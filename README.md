Golf Swing Error-to-Band (GPU-ready)
===================================

Modular Python 3.10+ pipeline for evaluating golf swings. The workflow runs video -> pose estimation -> keypoints -> features -> error classifier -> rule-based scoring -> band classification (1–10). GPU is preferred when available; CPU remains supported.

Allowed libraries: opencv-python, mediapipe, numpy, pandas, scikit-learn, matplotlib, tqdm, joblib, scipy (optional), pyyaml (optional).

Repository layout (Module 0)
----------------------------
- src/golf_pose: core package (config, devices, data, pose, features, models, scoring, reporting, io).
- scripts: CLI entrypoints for each module.
- outputs: keypoints, features, models, predictions, reports, logs (artifacts only; not tracked).

Modules (must run in order)
---------------------------
0) Skeleton (this commit).
1) Build Golf-Pose manifest.
2) Build BTC manifest.
3) Pose extraction (MediaPipe Pose, GPU preferred).
4) Feature engineering (normalized keypoints, biomechanical features).
5) Train error classifier on Golf-Pose.
6) BTC inference + scoring/band mapping.
7) Reporting (plots, stats, confusion matrix).

Quick start (after implementation)
----------------------------------
```bash
python scripts/module1_build_golf_pose_manifest.py --data-root <path_to_Golf-Pose>
python scripts/module2_build_btc_manifest.py --data-root <path_to_BTC>
python scripts/module3_extract_pose.py --manifest data/golf_pose_manifest.csv --device gpu
python scripts/module4_compute_features.py --keypoints-dir outputs/keypoints
python scripts/module5_train_error_classifier.py --features outputs/features/features.csv
python scripts/module6_run_btc_inference.py --features outputs/features/features.csv --device gpu
python scripts/module7_generate_reports.py --predictions outputs/predictions/btc_predictions.csv
```

Notes
-----
- No notebooks; all logic via scripts + package modules.
- Logging, argparse, and type hints are required in each module.
- MediaPipe GPU usage should be auto-selected when available, with CPU fallback.
- Datasets are not included in the repo; place them locally and reference via arguments.
- Golf-Pose manifest (Module 1) mặc định bỏ thư mục Good Swings; thêm `--include-good` nếu cần đưa vào.
