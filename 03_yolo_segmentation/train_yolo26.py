"""Train with the recovered run options. Use an existing local pretrained file."""

import argparse
from pathlib import Path
import yaml
from yolo_runtime import stage_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("training_config.yaml"))
    p.add_argument("--device", default="0")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    options = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    options.pop("model")
    if not a.weights.is_file():
        p.error("Supply an existing local pretrained checkpoint")
    if not a.execute:
        print(yaml.safe_dump(options, sort_keys=False))
        return
    cfg = stage_dataset(a.dataset, a.output, ("train", "valid", "test"))
    from ultralytics import YOLO
    YOLO(str(a.weights.resolve())).train(
        **options, data=str(cfg), device=a.device, project=str(a.output.resolve()), name="train",
    )


if __name__ == "__main__":
    main()
