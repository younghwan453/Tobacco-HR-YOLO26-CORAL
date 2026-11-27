# Roboflow polygons and export cleanup

The researcher records GPT Astra 6.2 + SAM3-assisted annotation in Roboflow followed by manual review. The retained [protocol](annotation_protocol.md), [rubric](hr_grade_definitions.md) and [QC notes](annotation_quality_control.md) describe that process. Exact Roboflow prompts and private run history are unavailable.

The filename utility is from 261122_colabFileNameChanger.ipynb, cell index 21. It removes the _jpg.rf.<export hash> suffix and optional corrected_ prefix, normalizes hyphens, and handles matching images and YOLO TXT labels in train/valid/test. It previews by default. With --output, it writes a new copy of the split folders and portable YAML when present. Filename collisions and labels without corresponding image stems raise errors.
