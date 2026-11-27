"""CSV and consensus rules from 261124sam3.ipynb."""
import csv
import re

import numpy as np

from normalize_image_ids import standard_id


CSV_SITE_ORDER = ("EV", "1", "2", "3", "4", "SARM1")
SITE_ORDER = ("SARM1", "4", "3", "EV", "1", "2")


def parse_triplet(values):
    if isinstance(values, str):
        v = re.split(r'[,;|\s]+', values.strip())
    else:
        v = [str(x).strip() for x in values]
    v = [x for x in v if x != '']
    if len(v) != 3:
        raise ValueError(f'3명 점수 형식 아님: {values!r}')
    vals = []
    for x in v:
        if x.lower() in ('na', 'nan', 'null', 'none', '-'):
            vals.append(None)
        else:
            x2 = float(x)
            if not x2.is_integer() or not 0 <= x2 <= 4:
                raise ValueError(f'HR 점수 범위(0-4) 벗어남: {x}')
            vals.append(int(x2))
    return vals


def parse_csv_row(row):
    if not row: return None
    key = standard_id(row[0])
    if not key: return None
    rest = row[1:]
    if len(rest) == 6:
        groups = [parse_triplet(x) for x in rest]
    elif len(rest) == 18:
        groups = [parse_triplet(rest[i:i+3]) for i in range(0, 18, 3)]
    else:
        raise ValueError(f'{key}: CSV 열 수 {len(row)} (7 또는 19열이어야 함)')
    return key, groups


def read_score_csv(path):
    err = None
    for enc in ['utf-8-sig', 'cp949']:
        try:
            result = {}
            with path.open('r', encoding=enc, newline='') as f:
                for raw in csv.reader(f):
                    rec = parse_csv_row(raw)
                    if rec is None: continue
                    k, groups = rec
                    if k in result:
                        raise ValueError(f'CSV에 이미지 ID 중복: {k}')
                    result[k] = groups
            return result
        except UnicodeDecodeError as e:
            err = e
    raise err or ValueError('CSV 불러오기 실패')


def median_hr(vals):
    if len(vals) != 3 or any(x is None for x in vals):
        return None
    if max(vals) - min(vals) >= 2:
        return None
    return int(np.median(vals))

def read_ratings_table(path):
    return {key: dict(zip(CSV_SITE_ORDER, triplets))
            for key, triplets in read_score_csv(path).items()}
