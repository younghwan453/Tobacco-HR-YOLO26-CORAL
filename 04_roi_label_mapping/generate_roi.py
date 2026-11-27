"""Leaf and infiltration bounding boxes from 261124sam3.ipynb."""
import cv2
import numpy as np
import pandas as pd

from load_rater_csv import SITE_ORDER
from normalize_image_ids import standard_id


SITE_GEOMETRY = {
    'SARM1': (0.19, 0.29, 0.39, 0.44),
    '4':     (0.50, 0.29, 0.39, 0.44),
    '3':     (0.81, 0.29, 0.39, 0.44),
    'EV':    (0.19, 0.72, 0.39, 0.44),
    '1':     (0.50, 0.72, 0.39, 0.44),
    '2':     (0.81, 0.72, 0.39, 0.44),
}


def load_overrides(path):
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'image','site','cx','cy','w','h'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: 필수 열 {required}')
    out = {}
    for _,r in df.iterrows():
        site = str(r['site']).strip()
        key = standard_id(r['image'])
        if site not in SITE_ORDER or not key: raise ValueError(f'잘못된 override 행: {r.to_dict()}')
        vals = tuple(float(r[k]) for k in ('cx','cy','w','h'))
        if not all(0 <= v <= 1 for v in vals): raise ValueError(f'override는 0~1: {r.to_dict()}')
        if (key, site) in out:
            raise ValueError(f'Duplicate ROI override: {key}/{site}')
        out[(key,site)] = vals
    return out


def get_leaf_bbox(image):
    h,w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat, value = hsv[:,:,1], hsv[:,:,2]
    # 흰 배경 제거. 일부 고사부는 채도가 낮아도 어둡다면 포함
    fg = np.uint8(((sat > 25) & (value < 250)) | (value < 155)) * 255
    kernel = np.ones((9,9), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError('잎 자동 검출 실패: 마스크 없음')
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.03*h*w:
        raise ValueError('잎 자동 검출 실패: 최대 객체가 너무 작음')
    x,y,bw,bh = cv2.boundingRect(cnt)
    # 약간의 여유를 줌
    pad_x = int(0.025*bw); pad_y = int(0.025*bh)
    return (max(0,x-pad_x), max(0,y-pad_y), min(w,x+bw+pad_x), min(h,y+bh+pad_y))


def make_rois(image, image_id, overrides=None):
    overrides = overrides or {}
    x0,y0,x1,y1 = get_leaf_bbox(image)
    W,H = image.shape[1], image.shape[0]
    lw,lh = x1-x0, y1-y0
    rois = {}
    for site in SITE_ORDER:
        cx,cy,bw,bh = overrides.get((image_id,site), SITE_GEOMETRY[site])
        l = int(round(x0 + (cx-bw/2)*lw))
        r = int(round(x0 + (cx+bw/2)*lw))
        t = int(round(y0 + (cy-bh/2)*lh))
        b = int(round(y0 + (cy+bh/2)*lh))
        box = (max(0,l),max(0,t),min(W,r),min(H,b))
        if box[0]>=box[2] or box[1]>=box[3]: raise ValueError(f'ROI 크기 오류: {site}')
        rois[site] = box
    return rois, (x0,y0,x1,y1)
