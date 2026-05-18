# Thực nghiệm ablation: đồ thị train-only vs full-log (H1)

## Mục tiêu

- **Train-only**: prerequisite/similarity được suy ra chỉ từ **tập train của từng fold** (`data/processed/<dataset>/fold_<f>/e_pre_train_only.csv`, `e_sim_train_only.csv`).
- **Full-log**: cùng một quy tắc suy luận nhưng trên **toàn bộ** parquet đã tiền xử lý (`data/processed/<dataset>/full_log/e_pre.csv`, `e_sim.csv`) — *cố ý* giống thực hành leaky (không dùng cho báo cáo protocol chính).

Baseline diagnostic **GKT/GIKT** chạy hai lần (train-only vs full-log) trên **cùng** learner split; kết quả gộp vào `results/tables/graph_ablation_summary.csv` và bảng LaTeX `results/tables/graph_ablation.tex`.

Trong `baseline_runner`, GKT/GIKT là **ensemble tuyến tính** (không phải PyTorch đầy đủ); nhánh ``graph`` làm mượt dự đoán theo **trung bình tỉ lệ đúng của các KC láng giềng** trên tập cạnh đã xuất. **Train-only**: láng giềng chỉ dùng nhãn trên **train fold** và cạnh từ `fold_<f>/`. **Full-log**: cạnh từ `full_log/` và láng giềng dùng tỉ lệ đúng KC **gộp trên toàn bộ parquet** (leaky có chỉnh ý ở nhánh graph). Trước đây chỉ đổi file cạnh mà vẫn dùng rate train cho láng giềng nên $\Delta$AUC/$\Delta$ACC thường gần 0.

## Điều kiện trước khi chạy

1. **Cài package** (để có CLI `p0-*` nếu cần):

   ```powershell
   pip install -e .
   ```

2. **Có parquet** cho cả ba benchmark:

   - `data/processed/junyi.parquet`
   - `data/processed/assist2012.parquet`
   - `data/processed/xes3g5m.parquet`

   Nếu thiếu, chạy preprocess trước (ví dụ):

   ```powershell
   python -m src.preprocess --config configs/junyi.yaml
   python -m src.preprocess --config configs/assist2012.yaml
   python -m src.preprocess --config configs/xes3g5m.yaml
   ```

   Hoặc pipeline đầy đủ: `scripts/run_all_datasets_full.ps1`.

3. **Cấu hình**: trong `configs/*.yaml` khối `graph_ablation` phải `enabled: true` và `models` liệt kê các model có nhánh graph trong `baseline_runner` (mặc định repo: `[gkt, gikt, skt, dygkt, dgekt]`).

## Chạy nhanh (Windows PowerShell)

Từ **thư mục gốc repo** (`p0_project`):

```powershell
.\scripts\run_graph_ablation_experiment.ps1 -ServerProfile
```

**Tiến trình hiển thị**

- PowerShell: thanh `Write-Progress` + dòng `[bước/tổng]` và % **theo từng lớp** (graph_builder → export → baseline → …); sau mỗi bước in thời gian đã chạy.
- Trong **baseline** (`baseline_runner`), log `Baseline progress [dataset] i/N (~p%)` cho từng cặp fold × model × `graph_construction` (đây là phần thường lâu nhất). Không có % training epoch bên trong PyTorch — nếu cần chi tiết hơn:  
  `python -m src.baseline_runner --config ... --log-level DEBUG`.

Tham số tùy chọn:

| Tham số | Ý nghĩa |
|--------|---------|
| `-ServerProfile` | Giới hạn thread BLAS (~8), giảm spike RAM |
| `-SkipGraphBuild` | Bỏ `graph_builder` nếu đã có fold_* |
| `-SkipFullLogExport` | Bỏ `export_full_log_graph` nếu đã có full_log/ |
| `-SkipBaseline` | Chỉ build/export đồ thị, không chạy baseline |

**Junyi** RAM lớn: script tự thêm `--skip-cold-start` cho baseline (vẫn tính **AUC/ACC/NLL đầy đủ** trên val+test; chỉ bỏ merge cold-start strata và giới hạn file prediction mẫu để tránh OOM).

## Chạy nhanh (Linux/macOS)

```bash
chmod +x scripts/run_graph_ablation_experiment.sh
SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh
```

## Chạy thủ công từng bước (để debug)

Với mỗi `configs/<dataset>.yaml`:

```powershell
python -m src.graph_builder --config configs/junyi.yaml
python -m src.export_full_log_graph --config configs/junyi.yaml
python -m src.baseline_runner --config configs/junyi.yaml --skip-cold-start
```

Assist / XES (không cần skip cold-start trừ khi máy yếu):

```powershell
python -m src.baseline_runner --config configs/assist2012.yaml
python -m src.baseline_runner --config configs/xes3g5m.yaml
```

Sinh lại bảng paper:

```powershell
python scripts/generate_paper_artifacts.py
```

## Artefact đầu ra

| File | Nội dung |
|------|----------|
| `results/tables/baseline_fold_results.csv` | Mọi fold × model × `graph_construction` |
| `results/tables/baseline_results.csv` | Mean + bootstrap CI; có cột `graph_construction` |
| `results/tables/graph_ablation_summary.csv` | Một dòng / dataset / model: AUC/ACC train-only vs full-log và Δ |
| `results/tables/graph_ablation.tex` | `\input` trong `paper/main.tex` |

Trong LaTeX, bảng baseline chính (`baseline_results.tex`) chỉ giữ **`train_only`** (do `generate_paper_artifacts.py` lọc).

## Ghi chú

- Full-log là **ablation có chủ đích**, không thay thế audit leakage-controlled.
- Nếu baseline Junyi vẫn OOM, đóng ứng dụng khác, bật `-ServerProfile`, hoặc tạm giảm `split.n_folds` trong YAML xuống `1` để thử pipeline (không khuyến nghị cho bài báo).
