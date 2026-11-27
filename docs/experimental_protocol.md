# Experimental protocol

1. Prepare images with the recovered PlantCV color/scale and crop commands, recording the image cohort and reference settings.
2. Follow the HR0–HR4 rubric and manually review the VLM-assisted Roboflow polygons. Preview export filename normalization, then write a matched, cleaned dataset to a new directory.
3. Check paired files and split overlap. Convert class indices to one region class while retaining the coordinates. Evaluate the recovered checkpoint against the recorded Crop dataset.
4. For the rater side workflow, use the original three-rater CSV, existing one-class polygons and bounding boxes. Keep unmatched masks, missing grades and ambiguous assignments in the review outputs.
5. For the 261127 CORAL experiment, use the original five-class GT masks and the 32 stored image features. Report GT validation separately from YOLO-mask inference. Keep predictions and results in new local directories.
