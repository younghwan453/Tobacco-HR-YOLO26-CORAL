# CORAL ordinal regression

The authoritative 261127coral.ipynb experiment trains on Roboflow five-class ground-truth polygons and applies the trained score head to YOLO polygons. The three-rater data and ROI audit are documented in Stage 4. The earlier 261124 notebook contributes the requested mapping workflow.

## Train and extract

The exact 32 features come from the notebook's shared GT/YOLO extractor. They include area, bounding-box geometry, compactness, solidity, normalized perimeter, RGB/HSV/Lab statistics, white/yellow/brown/green mask fractions, Canny edge density and Laplacian variance. Pixels are rasterized, and polygons below 20 mask pixels are skipped. The resulting train set has 391 rows and the validation set has 48.

The model is a shared score MLP (32→64→16→1) with ReLU, dropout 0.15 and ordered biases via softplus. It uses standardized training features, AdamW (LR 0.001, weight decay 0.001), cumulative BCE, seed 42, batch 64, 250 maximum epochs and patience 35. Class-balanced sampling is disabled in this run. The saved checkpoint selected epoch 237 on validation BCE.

train_coral_gt.py builds train and validation feature CSVs in a new directory. Add --train to perform fitting and write a checkpoint and validation results. The final test was disabled in the historical run.

## Evaluate and predict

evaluate_coral.py computes accuracy, grade MAE, expected-score MAE, QWK, a confusion matrix and per-class precision, recall, F1 and support. It can also load a trusted existing checkpoint with an existing feature table. Four ordered probabilities P(HR > k) define the predicted grade by counting those ≥0.5, and their sum yields a continuous expected score from 0 to 4.

predict_hr.py applies the same feature extractor and checkpoint to YOLO segmentation masks with image size 960, confidence 0.15 and IoU 0.70. It writes separate predictions and optional overlays. The archived run stored 46 YOLO polygon scores on eight validation images. The archived GT validation assessment uses 48 ground-truth polygons. [CORAL results](../results/coral_metrics.md)
