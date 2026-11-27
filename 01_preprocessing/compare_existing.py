"""Display the raw and corrected versions of a pre-existing image pair."""

import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("Use a .png output file")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, path, title in zip(axes, (args.raw, args.corrected), ("Raw", "Existing corrected")):
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    with args.output.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
