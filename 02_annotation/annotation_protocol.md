# Annotation protocol

The researcher reported importing color- and scale-adjusted leaf images into Roboflow, using GPT Astra 6.2 with SAM3 to assist grading and polygon generation, manually editing masks, and exporting YOLO Segmentation TXT for YOLO26. The class definitions are preserved verbatim in [hr_grade_definitions.md](hr_grade_definitions.md). Assistive model versions, invocation order, exact prompt, thresholds and edits per image require the Roboflow workspace history. The grade definitions provide the available specification for a future repeat of the annotation.

The local dataset YAML lists five numeric class names. Each non-empty TXT row consists of the class ID followed by normalized x/y vertex pairs. The original export contains 392 train, 48 validation and 48 test polygons across 70, 8 and 8 images, respectively.
