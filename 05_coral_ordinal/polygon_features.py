"""Ground-truth and YOLO mask feature extraction from 261127coral.ipynb."""
import cv2
import numpy as np
import pandas as pd

FEATURES = [
    "mask_area_fraction", "mask_bbox_fraction", "mask_width_fraction", "mask_height_fraction",
    "mask_aspect", "mask_compactness", "mask_solidity", "mask_perimeter_norm",
    "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std",
    "hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std",
    "lab_l_mean", "lab_a_mean", "lab_b_mean", "lab_l_std", "lab_a_std", "lab_b_std",
    "white_fraction", "yellow_fraction", "brown_fraction", "green_fraction",
    "edge_density", "texture_laplacian"
]
MIN_PIXELS = 20
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_bgr(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"이미지 읽기 실패: {path}")
    return bgr


def polygon_to_mask(pts, hw):
    h, w = hw
    a = np.asarray(pts, dtype=np.float32)
    mask = np.zeros((h,w), np.uint8)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 3:
        return mask.astype(bool)
    a = np.rint(a).astype(np.int32)
    a[:,0] = np.clip(a[:,0], 0, w-1)
    a[:,1] = np.clip(a[:,1], 0, h-1)
    cv2.fillPoly(mask, [a.reshape((-1,1,2))], 1)
    return mask.astype(bool)


def read_gt_masks(label_path, hw):
    h, w = hw
    result = []
    for ln, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(), start=1):
        if not line.strip():
            continue
        tokens = line.split()
        cls = int(tokens[0])
        coords = np.array([float(v) for v in tokens[1:]], dtype=np.float32)
        if (cls not in range(5) or len(coords) < 6 or len(coords) % 2
                or not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any()):
            raise ValueError(f"{label_path}:{ln} class/좌표 오류")
        pts = coords.reshape(-1,2) * np.asarray([w,h],dtype=np.float32)
        mask = polygon_to_mask(pts, hw)
        if int(mask.sum()) >= MIN_PIXELS:
            result.append((cls, mask))
    return result


def prepare_channels(bgr):
    return {
        'rgb': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        'hsv': cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
        'lab': cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
        'gray': cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
    }


def polygon_features(channels, mask):
    mask = mask.astype(bool)
    H,W = mask.shape
    area = int(mask.sum())
    if area < MIN_PIXELS:
        return None
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    perimeter = sum(float(cv2.arcLength(c, True)) for c in contours)
    biggest = max(contours, key=cv2.contourArea)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(biggest))), 1.0)
    ys,xs = np.where(mask)
    ww = int(xs.max()-xs.min()+1)
    hh = int(ys.max()-ys.min()+1)

    rgb = channels['rgb'][mask].astype(np.float32)/255.0
    hsv = channels['hsv'][mask].astype(np.float32)
    lab = channels['lab'][mask].astype(np.float32)/255.0
    hue,sat,val = hsv[:,0],hsv[:,1],hsv[:,2]
    white = (sat < 48) & (val > 184)
    yellow = (hue >= 13) & (hue <= 43) & (sat > 42) & (val > 86)
    brown = (hue >= 5) & (hue <= 25) & (sat > 45) & (val > 40) & (val < 186)
    green = (hue >= 33) & (hue <= 100) & (sat > 35) & (val > 50)

    if 'edges' not in channels:
        channels['edges'] = cv2.Canny(channels['gray'], 80, 160)
        channels['lap'] = cv2.Laplacian(channels['gray'], cv2.CV_32F, ksize=3)

    f = {
        'mask_area_fraction': area/float(H*W),
        'mask_bbox_fraction': (ww*hh)/float(H*W),
        'mask_width_fraction': ww/float(W),
        'mask_height_fraction': hh/float(H),
        'mask_aspect': ww/max(hh,1),
        'mask_compactness': 4*np.pi*area/max(perimeter**2,1.0),
        'mask_solidity': min(area/hull_area, 2.0),
        'mask_perimeter_norm': perimeter/max(np.hypot(H,W),1.0),
        'red_mean': rgb[:,0].mean(), 'green_mean': rgb[:,1].mean(), 'blue_mean': rgb[:,2].mean(),
        'red_std': rgb[:,0].std(), 'green_std': rgb[:,1].std(), 'blue_std': rgb[:,2].std(),
        'hue_mean': hue.mean()/179.0, 'hue_std': hue.std()/179.0,
        'sat_mean': sat.mean()/255.0, 'sat_std': sat.std()/255.0,
        'val_mean': val.mean()/255.0, 'val_std': val.std()/255.0,
        'lab_l_mean': lab[:,0].mean(), 'lab_a_mean': lab[:,1].mean(), 'lab_b_mean': lab[:,2].mean(),
        'lab_l_std': lab[:,0].std(), 'lab_a_std': lab[:,1].std(), 'lab_b_std': lab[:,2].std(),
        'white_fraction': white.mean(), 'yellow_fraction': yellow.mean(),
        'brown_fraction': brown.mean(), 'green_fraction': green.mean(),
        'edge_density': (channels['edges'][mask] > 0).mean(),
        'texture_laplacian': min(float(channels['lap'][mask].var())/10000,10.0)
    }
    return {k:float(f[k]) for k in FEATURES}


def pairs_for(root, split):
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"{split}: {images_dir} 또는 {labels_dir} 없음")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"{split}: 이미지 파일 없음")
    names = [p.stem.lower() for p in images]
    if len(names) != len(set(names)):
        raise ValueError(f"{split}: 중복 이미지 기본 이름이 존재합니다")
    missing = [p.name for p in images if not (labels_dir / (p.stem + '.txt')).is_file()]
    if missing:
        raise FileNotFoundError(f"{split}: 이미지와 TXT 파일 이름이 맞지 않음: {missing[:8]}")
    return [(img, labels_dir / (img.stem + ".txt")) for img in images]


def extract_gt_features(split, pairs):
    rows=[]
    for idx,(image_path,txt_path) in enumerate(pairs,1):
        bgr=read_bgr(image_path)
        ch=prepare_channels(bgr)
        gt=read_gt_masks(txt_path, bgr.shape[:2])
        for polygon_id,(hr,mask) in enumerate(gt):
            f=polygon_features(ch,mask)
            if f is None:
                continue
            row={'split':split,'image':image_path.name,'polygon_id':polygon_id,'hr_grade':int(hr)}
            row.update(f)
            rows.append(row)
        if idx%20==0 or idx==len(pairs):
            print(f'{split}: {idx}/{len(pairs)} 이미지 처리')
    df=pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'{split}: 유효 GT Polygon이 없습니다')
    if not np.isfinite(df[FEATURES].to_numpy(np.float64)).all():
        raise ValueError(f'{split}: 특징에 NaN/Inf 값이 있습니다')
    return df
