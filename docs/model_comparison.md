# Interpreting metrics

The independent YOLO evaluation measures localization of the one-class HR_region over eight Crop test images. The CORAL score is evaluated on 48 Roboflow validation polygons. The separate three-rater workflow audits per-site grade linkage for existing polygons. These evaluation units and supervision sources should be reported with each result.

A full integrated five-class instance-segmentation evaluation requires locked independent test images, original HR0–HR4 GT polygons and predicted masks carrying CORAL grades and confidence. Report per-class AP across mask IoU thresholds. The 261127 run leaves this evaluation pending.
