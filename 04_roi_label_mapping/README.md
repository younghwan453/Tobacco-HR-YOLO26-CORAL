# Bounding boxes, polygon overlap and rater CSV

The adapted source is 261124sam3.ipynb cells 36–40 and its subsequent CSV audit. It takes existing segmentation polygons and never requires SAM3 installation, GPT, Gemini, or service credentials.

The source log verified the CSV order EV, 1, 2, 3, 4, SARM1 against eight prior audit images (48/48 matches and a unique best ordering). An image ID may use hyphens or underscores. Score rows accept six packed triplets or 18 expanded score columns. The consensus is the median when all three grades are present and their spread is less than 2. Missing grades and larger disagreement produce NA. The source output records 86 images and 516 site rows, with 514 matching grade values and two matching NA values after comparison against the original CSV. The mapping cell reports zero ROI overrides.

The algorithm segments the leaf foreground with the recovered HSV/value thresholds, selects its largest contour and computes a proportional bounding box for each site. The spatial order is SARM1, 4, 3, then EV, 1, 2. The code accepts an optional ROI override CSV with image, site, cx, cy, w and h, in fractions of the detected leaf box.

The mapper rasterizes each existing polygon and divides the number of mask pixels inside each ROI by the full polygon area. It proposes the best ROI at overlap ≥0.35. If a second ROI reaches both 0.30 and 80% of the best overlap, the proposal is ambiguous. The historical candidate export flagged ambiguous proposals for review. This repository preserves their proposed site and score and withholds automatic grade assignment. Missing polygons also remain missing. A reviewed site can have multiple polygons. The optional overlay marks the box and retained or review polygons. No candidate or synthetic grade is fed into the 261127 CORAL run.

Run validate_labels.py on the existing one-class Crop dataset and the original rater CSV. It reports aggregate counts without writing by default. --output writes per-site and per-polygon audit CSVs, with --previews to save overlays locally. [Rechecked aggregate counts](../results/roi_mapping_aggregate.json)
