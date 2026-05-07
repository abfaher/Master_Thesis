import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path("/workspace/LLVIP")
ANN = BASE / "Annotations"
SPLITS = [
    ("infrared/train/images", "infrared/train/labels"),
    ("infrared/test/images",  "infrared/test/labels"),
]
IMG_EXTS = {".jpg", ".jpeg"}
CLASS_NAME = "person"  # class 0

def find_xml_for_stem(stem: str) -> Path | None:
    """Return the first existing <stem>.xml under ANN, else None."""
    p = ANN / f"{stem}.xml"
    if p.exists():
        return p
    return None

def yolo_line(xmin, ymin, xmax, ymax, w, h):
    """
    This function turns one VOC-style box (xmin, ymin, xmax, ymax in pixels) 
    into a single YOLO label line ("0 cx cy w h" in [0,1] normalized coords; stored as a
    fraction of the image)
    """
    xmin = max(0.0, min(float(xmin), w))
    xmax = max(0.0, min(float(xmax), w))
    ymin = max(0.0, min(float(ymin), h))
    ymax = max(0.0, min(float(ymax), h))
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    if bw <= 0 or bh <= 0:
        return None
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw /= w; bh /= h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"  # class 0 = person

def convert_one(xml_path: Path) -> list[str]:
    """Parse one XML and return YOLO lines for class 'person'."""
    root = ET.parse(xml_path).getroot()

    # image size
    W = float(root.findtext("size/width"))
    H = float(root.findtext("size/height"))

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != CLASS_NAME:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        line = yolo_line(
            bb.findtext("xmin"), bb.findtext("ymin"),
            bb.findtext("xmax"), bb.findtext("ymax"),
            W, H
        )
        if line:
            lines.append(line)
    return lines

def process_split(img_dir: Path, lbl_dir: Path) -> tuple[int, int, int]:
    """Return (images_seen, labels_written, images_missing_xml)."""
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images_seen = 0
    labels_written = 0
    missing_xml = 0

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        images_seen += 1
        stem = img_path.stem
        xml_path = find_xml_for_stem(stem)

        if xml_path is None:
            # No annotation found -> still write an empty label file so YOLO won't complain
            (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            missing_xml += 1
            continue

        lines = convert_one(xml_path)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        labels_written += 1

    return images_seen, labels_written, missing_xml

def main():
    totals = {"imgs": 0, "lbls": 0, "miss": 0}
    for img_rel, lbl_rel in SPLITS:
        img_dir = BASE / img_rel
        lbl_dir = BASE / lbl_rel
        if not img_dir.exists():
            print(f"Skip: {img_dir} not found.")
            continue
        imgs, lbls, miss = process_split(img_dir, lbl_dir)
        totals["imgs"] += imgs
        totals["lbls"] += lbls
        totals["miss"] += miss
        print(f"[{img_rel}] images={imgs}, label_files_written={lbls}, missing_xml={miss}")
        print(f"  labels -> {lbl_dir}")

    print("\nDone.")
    print(f"TOTAL: images={totals['imgs']}, label_files_written={totals['lbls']}, missing_xml={totals['miss']}")

if __name__ == "__main__":
    main()
