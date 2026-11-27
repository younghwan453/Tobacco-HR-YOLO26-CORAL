"""Apply the archived 261127 CORAL checkpoint to YOLO polygon features."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from coral_model import load_checkpoint, predict_coral_from_features
from polygon_features import FEATURES, polygon_features, polygon_to_mask, prepare_channels, read_bgr


IMG_SIZE = 960
YOLO_CONF = 0.15
YOLO_NMS_IOU = 0.70
GRADE_COLORS_BGR=[(120,180,100),(210,190,90),(80,200,240),(50,140,240),(70,70,230)]


def predict_yolo_to_coral(image_path, seg, model, scaler, device, output, save_overlay=True):
    image_path=Path(image_path)
    bgr=read_bgr(image_path)
    H,W=bgr.shape[:2]
    result=seg.predict(source=str(image_path),imgsz=IMG_SIZE,conf=YOLO_CONF,
                       iou=YOLO_NMS_IOU,device=str(device),verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return pd.DataFrame(columns=['image','polygon_id','yolo_conf','hr_pred','hr_expected'])
    polys=result.masks.xy
    confs=result.boxes.conf.detach().cpu().numpy()
    ch=prepare_channels(bgr)
    rows=[]
    for i,(polygon,confidence) in enumerate(zip(polys,confs)):
        mask=polygon_to_mask(polygon,(H,W))
        f=polygon_features(ch,mask)
        if f is None:
            continue
        row={'image':image_path.name,'polygon_id':i,'yolo_conf':float(confidence),
             'mask_area_pixels':int(mask.sum())}
        row.update(f)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['image','polygon_id','yolo_conf','hr_pred','hr_expected'])
    df=predict_coral_from_features(pd.DataFrame(rows), model, scaler, device)

    if save_overlay:
        overlay=bgr.copy()
        for row in df.itertuples(index=False):
            poly=np.asarray(polys[int(row.polygon_id)],dtype=np.int32)
            if poly.ndim!=2 or len(poly)<3:
                continue
            mask=polygon_to_mask(poly,(H,W))
            color=GRADE_COLORS_BGR[int(row.hr_pred)]
            overlay[mask]=(0.68*overlay[mask].astype(np.float32)+0.32*np.array(color)).astype(np.uint8)
            cv2.polylines(overlay,[poly.reshape(-1,1,2)],True,color,2)
            xy=poly.mean(axis=0).astype(int)
            x,y=int(np.clip(xy[0],0,W-1)),int(np.clip(xy[1],22,H-5))
            cv2.putText(overlay,f'HR{int(row.hr_pred)} {row.hr_expected:.2f}',(x,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(overlay,f'HR{int(row.hr_pred)} {row.hr_expected:.2f}',(x,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1,cv2.LINE_AA)
        dest=output/'overlays'
        dest.mkdir(exist_ok=True)
        outpath=dest/f'{image_path.stem}_CORAL.png'
        ok, data = cv2.imencode('.png', overlay)
        if not ok:
            raise OSError(f'Cannot encode {outpath}')
        data.tofile(str(outpath))
    return df.drop(columns=FEATURES)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--yolo", type=Path, required=True)
    p.add_argument("--coral", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-overlay", action="store_true")
    a = p.parse_args()
    if not all(path.is_file() for path in [*a.images, a.yolo, a.coral]):
        p.error("Supply existing trusted checkpoint and image files")
    if len({path.stem.casefold() for path in a.images}) != len(a.images):
        p.error("Input images must have unique stems")
    a.output.mkdir(parents=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(a.output.resolve() / "ultralytics"))
    from ultralytics import YOLO
    seg = YOLO(str(a.yolo.resolve()))
    device = torch.device(a.device)
    model, scaler = load_checkpoint(a.coral, device)
    frames = [predict_yolo_to_coral(image, seg, model, scaler, device, a.output, not a.no_overlay)
              for image in a.images]
    pd.concat(frames, ignore_index=True).to_csv(
        a.output / "yolo_coral_predictions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
