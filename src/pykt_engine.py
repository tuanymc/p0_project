"""Train/evaluate neural baselines via pyKT with P0 protocol graphs (optional dependency)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _mean_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-4) -> float:
    y = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _batch_to_device(dcur: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dcur.items()}


def _model_forward_loss(model, batch: dict, model_name: str) -> torch.Tensor:
    from torch.nn.functional import binary_cross_entropy, one_hot

    device = next(model.parameters()).device
    dcur = _batch_to_device(batch, device)
    q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
    qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
    sm = dcur["smasks"]
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if model_name == "dkt":
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return binary_cross_entropy(pred.double(), target.double())
    if model_name == "akt":
        y, reg = model(cc.long(), cr.long(), cq.long())
        y = y[:, 1:]
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return binary_cross_entropy(pred.double(), target.double()) + reg
    if model_name == "gkt":
        y = model(cc.long(), cr.long())
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return binary_cross_entropy(pred.double(), target.double())
    if model_name == "simplekt":
        y, _y2, _y3 = model(dcur, train=True)
        y = y[:, 1:]
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return binary_cross_entropy(pred.double(), target.double())
    raise ValueError(f"Unsupported model_name={model_name}")


def _evaluate_detailed(model, loader, model_name: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    from pykt.models.evaluate_model import device as pt_device

    model.eval()
    y_trues: list[np.ndarray] = []
    y_scores: list[np.ndarray] = []
    dev = torch.device(pt_device)
    from torch.nn.functional import one_hot

    with torch.no_grad():
        for data in loader:
            dcur = _batch_to_device(data, dev)
            q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
            qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
            sm = dcur["smasks"]

            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)

            if model_name == "dkt":
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "akt":
                y, _reg = model(cc.long(), cr.long(), cq.long())
                y = y[:, 1:]
            elif model_name == "simplekt":
                preds = model(dcur, train=False)
                y = preds[:, 1:]
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
            else:
                raise ValueError(f"Unsupported pyKT model_name={model_name}")

            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach().cpu()
            y_trues.append(t.numpy())
            y_scores.append(y.numpy())

    ts = np.concatenate(y_trues, axis=0)
    ps = np.concatenate(y_scores, axis=0)
    if len(np.unique(ts)) < 2:
        auc = float("nan")
    else:
        auc = float(metrics.roc_auc_score(ts, ps))
    acc = float(metrics.accuracy_score(ts, (ps >= 0.5).astype(int)))
    return auc, acc, ts, ps


def _train_loop(model, train_loader, valid_loader, epochs: int, lr: float, patience: int = 8) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = -1.0
    stale = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for data in train_loader:
            loss = _model_forward_loss(model, data, model.model_name)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        tr_loss = float(np.mean(losses)) if losses else 0.0
        auc, acc, _, _ = _evaluate_detailed(model, valid_loader, model.model_name)
        logger.info("pyKT epoch %s train_loss=%.5f valid_auc=%.5f valid_acc=%.5f", ep, tr_loss, auc, acc)
        if auc > best_auc + 1e-4:
            best_auc = auc
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)


def run_pykt_fold(
    *,
    display_model: str,
    pykt_name: str,
    work_dir: Path,
    num_q: int,
    num_c: int,
    graph_npz: Path | None,
    graph_tag: str,
    hyperparams: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    max_seq_len: int,
) -> tuple[float, float, float, str]:
    """Train on fold 0 / validate on fold 1 rows inside ``train_valid_sequences.csv``; eval fold -1 test file."""
    import shutil

    from pykt.datasets.data_loader import KTDataset
    from pykt.models.init_model import init_model

    work_dir.mkdir(parents=True, exist_ok=True)
    for p in work_dir.glob("*.pkl"):
        try:
            p.unlink()
        except OSError:
            pass

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_type = ["questions", "concepts"]
    tv_path = str(work_dir / "train_valid_sequences.csv")
    te_path = str(work_dir / "test_sequences.csv")

    train_ds = KTDataset(tv_path, input_type, {0})
    valid_ds = KTDataset(tv_path, input_type, {1})
    eval_ds = KTDataset(te_path, input_type, {-1})

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    emb_type = "qid"
    data_cfg = {
        "dpath": str(work_dir),
        "num_q": int(num_q),
        "num_c": int(num_c),
        "emb_path": "",
        "train_valid_original_file": "train_valid_sequences.csv",
        "test_original_file": "test_sequences.csv",
    }

    if pykt_name == "dkt":
        emb_size = int(hyperparams.get("emb_size", hyperparams.get("hidden_dim", 100)))
        dropout = float(hyperparams.get("dropout", 0.2))
        model_cfg = {"emb_size": emb_size, "dropout": dropout}
    elif pykt_name == "akt":
        model_cfg = {
            "d_model": int(hyperparams.get("d_model", 256)),
            "n_blocks": int(hyperparams.get("n_blocks", 1)),
            "dropout": float(hyperparams.get("dropout", 0.05)),
            "d_ff": int(hyperparams.get("d_ff", 1024)),
            "kq_same": int(hyperparams.get("kq_same", 1)),
            "final_fc_dim": int(hyperparams.get("final_fc_dim", 512)),
            "num_attn_heads": int(hyperparams.get("num_attn_heads", 8)),
            "separate_qa": bool(hyperparams.get("separate_qa", False)),
            "l2": float(hyperparams.get("l2", 1e-5)),
        }
    elif pykt_name == "gkt":
        if graph_npz is None:
            raise ValueError("GKT requires graph_npz")
        shutil.copy(graph_npz, work_dir / f"gkt_graph_{graph_tag}.npz")
        model_cfg = {
            "hidden_dim": int(hyperparams.get("hidden_dim", 100)),
            "emb_size": int(hyperparams.get("emb_size", 100)),
            "dropout": float(hyperparams.get("dropout", 0.5)),
            "graph_type": graph_tag,
        }
    elif pykt_name == "simplekt":
        model_cfg = {
            "d_model": int(hyperparams.get("d_model", 256)),
            "n_blocks": int(hyperparams.get("n_blocks", 4)),
            "dropout": float(hyperparams.get("dropout", 0.05)),
            "d_ff": int(hyperparams.get("d_ff", 1024)),
            "num_attn_heads": int(hyperparams.get("num_attn_heads", 8)),
            "kq_same": int(hyperparams.get("kq_same", 1)),
            "l2": float(hyperparams.get("l2", 1e-4)),
            "seq_len": int(max_seq_len),
            "final_fc_dim": int(hyperparams.get("final_fc_dim", 512)),
            "final_fc_dim2": int(hyperparams.get("final_fc_dim2", 256)),
            "separate_qa": bool(hyperparams.get("separate_qa", False)),
        }
    else:
        raise ValueError(f"Unknown pyKT name={pykt_name}")

    model = init_model(pykt_name, model_cfg, data_cfg, emb_type)
    if model is None:
        raise RuntimeError(f"pyKT init_model returned None for {pykt_name}")

    note = (
        f"pyKT `{pykt_name}` trained on learner-split train users; metrics on valid+test sequence positions. "
        "GKT adjacency from P0 exported graphs."
    )
    if display_model != pykt_name:
        note += f" YAML alias `{display_model}` maps to `{pykt_name}` (GIKT not bundled in pyKT)."

    patience = int(hyperparams.get("patience", 8))
    _train_loop(model, train_loader, valid_loader, epochs=max(1, int(epochs)), lr=float(lr), patience=patience)
    auc, acc, ts, ps = _evaluate_detailed(model, eval_loader, model.model_name)
    nll = _mean_nll(ts, ps)
    return auc, acc, nll, note
