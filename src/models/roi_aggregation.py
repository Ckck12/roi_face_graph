# src/models/roi_aggregation.py

import torch

def compute_overlap_ratio(bbox, patch_box):
    """
    bbox: (x1, y1, x2, y2)
    patch_box: (x1, y1, x2, y2)
    return overlap / area(patch_box)
    """
    x1 = max(bbox[0], patch_box[0])
    y1 = max(bbox[1], patch_box[1])
    x2 = min(bbox[2], patch_box[2])
    y2 = min(bbox[3], patch_box[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    overlap = w * h

    patch_area = (patch_box[2] - patch_box[0]) * (patch_box[3] - patch_box[1])
    if patch_area <= 0:
        return 0.0
    return overlap / patch_area
