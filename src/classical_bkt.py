"""Classical multi-skill BKT (independent per KC); scipy-fitted parameters."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class BKTParams:
    p_init: float
    p_learn: float
    p_guess: float
    p_slip: float


def _skill_sequences(train_df: pd.DataFrame, skill_col: str = "kc_id") -> dict[int, list[np.ndarray]]:
    """Per learner, temporal outcome sequences on each practiced KC."""
    per_skill: dict[int, list[np.ndarray]] = defaultdict(list)
    for _, grp in train_df.groupby("user_id", sort=False):
        g = grp.sort_values("timestamp")
        for k in g[skill_col].unique():
            sub = g[g[skill_col] == int(k)]["correct"].to_numpy(dtype=np.int8)
            if len(sub) >= 4:
                per_skill[int(k)].append(sub)
    return per_skill


def _forward_student_ll(obs: np.ndarray, p: BKTParams) -> float:
    eps = 1e-9
    pi_m = float(np.clip(p.p_init, eps, 1 - eps))
    ll = 0.0
    for y in obs.astype(int):
        py = (1 - pi_m) * (p.p_guess if y == 1 else (1 - p.p_guess)) + pi_m * (
            (1 - p.p_slip) if y == 1 else p.p_slip
        )
        py = max(py, eps)
        ll += float(np.log(py))
        num = pi_m * ((1 - p.p_slip) if y == 1 else p.p_slip)
        post_m = num / py
        pi_m = post_m + (1 - post_m) * p.p_learn
        pi_m = float(np.clip(pi_m, eps, 1 - eps))
    return ll


def _fit_skill(observations: list[np.ndarray]) -> BKTParams | None:
    if not observations:
        return None
    flat_len = sum(len(o) for o in observations)
    if flat_len < 30:
        return None

    def neg_ll(theta: np.ndarray) -> float:
        if np.any(theta <= 1e-4) or np.any(theta >= 1 - 1e-4):
            return 1e12
        prm = BKTParams(float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]))
        total = 0.0
        for obs in observations:
            total -= _forward_student_ll(obs, prm)
        return float(total)

    x0 = np.array([0.25, 0.25, 0.2, 0.2], dtype=np.float64)
    bounds = [(0.05, 0.95)] * 4
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 120})
    if not res.success:
        logger.warning("BKT fit warning: %s", res.message)
    th = np.clip(res.x, 1e-3, 1 - 1e-3)
    return BKTParams(float(th[0]), float(th[1]), float(th[2]), float(th[3]))


class ClassicalBKTMultiSkill:
    """Independent BKT parameters per integer KC id (original parquet ids)."""

    def __init__(self, params: dict[int, BKTParams], marginal_rate: float) -> None:
        self.params = params
        self.marginal_rate = marginal_rate

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> ClassicalBKTMultiSkill:
        runs = _skill_sequences(train_df)
        fitted: dict[int, BKTParams] = {}
        for k, seqs in runs.items():
            pr = _fit_skill(seqs)
            if pr is not None:
                fitted[k] = pr
        mr = float(train_df["correct"].mean()) if len(train_df) else 0.5
        logger.info("Classical BKT: fitted %d skills (eligible sequences)", len(fitted))
        return cls(fitted, mr)

    def _belief_step(self, pi_m: float, y: int, pr: BKTParams) -> float:
        eps = 1e-9
        pi_m = float(np.clip(pi_m, eps, 1 - eps))
        py = (1 - pi_m) * (pr.p_guess if y == 1 else (1 - pr.p_guess)) + pi_m * (
            (1 - pr.p_slip) if y == 1 else pr.p_slip
        )
        py = max(py, eps)
        num = pi_m * ((1 - pr.p_slip) if y == 1 else pr.p_slip)
        post_m = num / py
        return float(np.clip(post_m + (1 - post_m) * pr.p_learn, eps, 1 - eps))

    def predict_probabilities(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        beliefs: dict[tuple[int, int], float] = {}

        def init_pi(kc: int) -> float:
            pr = self.params.get(kc)
            return pr.p_init if pr is not None else self.marginal_rate

        train_sorted = train_df.sort_values(["user_id", "timestamp"])
        for _, row in train_sorted.iterrows():
            uid = int(row["user_id"])
            kc = int(row["kc_id"])
            y = int(row["correct"])
            key = (uid, kc)
            pi = beliefs.get(key, init_pi(kc))
            pr = self.params.get(kc)
            if pr is None:
                beliefs[key] = float(np.clip(pi * 0.9 + y * 0.1, 1e-3, 1 - 1e-3))
            else:
                beliefs[key] = self._belief_step(pi, y, pr)

        eval_reset = eval_df.reset_index(drop=True)
        order = np.lexsort((eval_reset["timestamp"].to_numpy(), eval_reset["user_id"].to_numpy()))
        probs = np.zeros(len(eval_reset), dtype=np.float64)
        for i in order:
            row = eval_reset.iloc[int(i)]
            uid = int(row["user_id"])
            kc = int(row["kc_id"])
            y = int(row["correct"])
            key = (uid, kc)
            pi = beliefs.get(key, init_pi(kc))
            pr = self.params.get(kc)
            if pr is None:
                probs[int(i)] = self.marginal_rate
                beliefs[key] = float(np.clip(pi * 0.9 + y * 0.1, 1e-3, 1 - 1e-3))
            else:
                probs[int(i)] = (1 - pi) * pr.p_guess + pi * (1 - pr.p_slip)
                beliefs[key] = self._belief_step(pi, y, pr)
        return probs
