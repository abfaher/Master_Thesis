# file containing the methods we will use for the fusion

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_xyxy(a, b):
    """Compute the Intersection over Union (IoU) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # compute the intersection angles
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    # compute intersection width & height
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    
    inter = iw * ih # intersection area: if inter == 0 -> no overlap
    if inter <= 0:
        return 0.0

    # compute the areas of the two bounding boxes
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # compute the union U
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def clamp_box_xyxy(b):
    """Safety: ensure x2>=x1 and y2>=y1 (avoid negative widths/heights)."""
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]