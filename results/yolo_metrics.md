# One-class YOLO26-seg results

| Evidence | Split | Mask precision | Mask recall | Mask mAP50 | Mask mAP50–95 |
|---|---|---:|---:|---:|---:|
| 261127coral.ipynb independent evaluation, 8 images and 49 polygons | Test | 0.911 | 0.834 | 0.8941203 | 0.7555295 |
| 261127coral.ipynb best-checkpoint validation, 8 images and 46 polygons | Validation | 0.993 | 0.870 | 0.913 | 0.755 |

Test AP comes from the explicit mask metric scalars. Precision and recall are rounded as shown in the notebook. The checkpoint is the best model from the run under runs_HR_1class/yolo26n_mild_aug_v1. Its notebook and saved args identify the existing Crop dataset. Both validation and test values measure the single-class HR_region model.
