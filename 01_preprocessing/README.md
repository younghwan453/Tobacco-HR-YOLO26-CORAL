# PlantCV preprocessing

Source notebook 261122Colab_ImageAdjustment.ipynb. The main batch normalization is cell index 14 and the crop routine with logged output is index 17 (zero-based). Later single-image and duplicate parameter trials were study material.

## Color and scale

The recorded batch settings are TARGET_MM_PER_PX = 0.1, a nominal chip width of 40.0 mm, and target area (40.0 / 0.1)² = 160,000 pixels². The 40 mm value is an input assumption in the source notebook. A physical measurement of that particular chart and an output log tying the normalization cell to each final image were unavailable. The adjacent 0.20 mm/px example is commented out.

For each JPG, reset PlantCV outputs, run auto_correct_color with radius 15 and read median_color_chip_size. Resize the corrected image by sqrt(target chip area / observed median chip area), using OpenCV INTER_AREA below scale 1 and INTER_CUBIC otherwise. The stored Colab output identifies PlantCV 4.11.3 and OpenCV 4.11.0.86. The user reported a Colab T4 runtime. The PlantCV, NumPy and OpenCV operations here use CPU APIs.

## Crop

The crop cell logs the leaf_on_white mode on a validation folder. It detects the ColorChecker, cuts the image at the left edge of the chart minus max(15 pixels, 2.5% of width), and preserves the white paper around the leaf. Its fallback uses grayscale threshold 65 and a qualifying contour on the right side. White-sheet bounds are detected with threshold 160 and 21×21 morphological filters. Leaf detection uses Lab chroma >14 or HSV saturation >45, with border exclusion and morphological cleanup. The leaf bounding box is padded by 12% with a 25-pixel minimum. Coordinates therefore vary by image. The logged inputs were 4000×3000 and the eight crops ranged from 1390–1655 px wide and 1170–1421 px high.

Run normalize and crop separately with explicit input and new output directories. The crop accepts leaf_on_white, white_sheet, or remove_card_only. The comparison utility displays a pair of existing images without recalibrating them.
