import pathlib

ROOT = pathlib.Path("/workspace/LLVIP")

def count_missing(list_file):
    n_img = n_ok = 0
    missing = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            p = pathlib.Path(line.strip())
            if not p.exists():
                continue
            n_img += 1
            # YOLO expects labels under <split>/labels/<same_basename>.txt
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            if lbl.exists():
                n_ok += 1
            else:
                missing.append((p, lbl))
    return n_img, n_ok, missing

for name in ["llvip_ir_train.txt", "llvip_ir_val.txt", "llvip_ir_test.txt"]:
    lf = ROOT / name
    if lf.exists():
        n_img, n_ok, missing = count_missing(lf)
        print(f"{name}: images={n_img}, labels_found={n_ok}, missing={n_img - n_ok}")
        for p, lbl in missing[:5]:
            print("  example missing:", p, "expected:", lbl)
    else:
        print(f"Not found: {lf}")
