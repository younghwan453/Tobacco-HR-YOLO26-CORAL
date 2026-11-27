# One-class YOLO26 segmentation

The one-class dataset includes healthy HR0 and other annotated infiltration regions. The converter changes IDs 0–4 to HR_region (ID 0) while preserving polygon coordinates. All 86 corresponding label files in the supplied five-class original and one-class conversion have identical coordinate tokens.

261127coral.ipynb records both dataset preparation and training. Keep the two data artifacts distinct in reproductions. The converted export contains 392/48/48 polygons over 70/8/8 images. The captured Windows training and independent test log for the named checkpoint shows the existing Crop dataset, with 284/46/49 polygons over 70/8/8 images. The captured args.yaml agrees with that data path. The notebook additionally contains an earlier training cell for the Crop lineage with a different run directory.

The recovered recipe is YOLO26n-seg pretrained, image size 960, batch 4, AdamW, initial LR 0.0005, cosine scheduling, final LR multiplier 0.01, and seed 42. Its 100-epoch budget stopped after epoch 56, with the best model at epoch 16 and patience 40. Geometric augmentation uses ±3° rotation, 0.03 translation, 0.10 scaling and 0.50 left/right flip. HSV, mosaic, mixup, cutmix and copy-paste were set to zero. The run uses RTX 5080, PyTorch 2.11.0+cu128 and Ultralytics 8.4.219.

The training CLI defaults to dry run. Use --execute to train when desired. The evaluation CLI stages just the requested split in a new output directory and lets the existing checkpoint and Ultralytics choose image size and batch unless overridden. The notebook uses this form for the independent test. [Metrics](../results/yolo_metrics.md)
