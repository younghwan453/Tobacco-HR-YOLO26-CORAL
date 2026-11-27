"""Image ID normalization from 261124sam3.ipynb."""
import re


def standard_id(raw):
    text = str(raw).upper().replace('-', '_')
    m = re.search(r'(SHA|SHB|YHA|YHB)_\d+_\d+_\d+', text)
    return m.group(0) if m else None


canonical_id = standard_id
