"""Pixel overlap and union routines from 261124sam3.ipynb."""
import cv2
import numpy as np


def load_yolo_polygons(label_path, shape):
    h,w = shape[:2]
    if not label_path.exists(): return []
    polys = []
    for lineno,line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        pieces = line.split()
        if float(pieces[0]) != 0:
            raise ValueError(f'단일 클래스(0) 아님: {label_path}:{lineno}')
        coords = np.asarray([float(s) for s in pieces[1:]], dtype=float)
        if len(coords)<6 or len(coords)%2: raise ValueError(f'Polygon 좌표 이상: {label_path}:{lineno}')
        if not np.isfinite(coords).all() or (coords < -0.001).any() or (coords > 1.001).any():
            raise ValueError(f'정규화 좌표 범위 이상: {label_path}:{lineno}')
        pts = coords.reshape(-1,2)
        pts[:,0] = np.clip(pts[:,0]*w, 0, w-1)
        pts[:,1] = np.clip(pts[:,1]*h, 0, h-1)
        polys.append(pts.astype(np.int32))
    return polys


def assign_roi(poly, rois, shape, min_overlap=0.35):
    h,w = shape[:2]
    mask = np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1,1,2)], 1)
    total = int(mask.sum())
    if not total: return None, 0, False
    rankings = []
    for site,(x0,y0,x1,y1) in rois.items():
        frac = int(mask[y0:y1,x0:x1].sum()) / total
        rankings.append((frac,site))
    rankings.sort(reverse=True)
    top, site = rankings[0]
    second = rankings[1][0] if len(rankings) > 1 else 0.0
    ambiguous = (top >= min_overlap and second >= 0.80*top and second >= 0.30)
    if top < min_overlap: return None, top, ambiguous
    return site, top, ambiguous


def polygon_union_to_contours(poly_list, shape, min_area=15):
    h,w = shape[:2]
    mask = np.zeros((h,w),np.uint8)
    for poly in poly_list:
        cv2.fillPoly(mask,[poly.reshape(-1,1,2)],255)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area: continue
        epsilon = 0.001*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1,2)
        if len(approx) >= 3: out.append(approx)
    return out, mask
