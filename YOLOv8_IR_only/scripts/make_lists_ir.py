import pathlib, random

ROOT = pathlib.Path("/workspace/LLVIP")
IR_TRAIN = ROOT/"infrared"/"train"/"images"
IR_TEST  = ROOT/"infrared"/"test"/"images"

def list_images(folder):
    exts = {".jpg"}
    return sorted([p.resolve() for p in folder.rglob("*.*") if p.suffix.lower() in exts])

random.seed(0)
train_imgs = list_images(IR_TRAIN)
test_imgs  = list_images(IR_TEST)

idx = list(range(len(train_imgs))); random.shuffle(idx)
n_val = max(1, int(0.15*len(idx)))   # 15% of IR/train → val
val_idx = set(idx[:n_val]); train_idx = set(idx[n_val:])

def write_list(paths, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p.as_posix() + "\n")  # ensure forward slashes since we're using Docker now

write_list([train_imgs[i] for i in sorted(train_idx)], ROOT/"llvip_ir_train.txt")
write_list([train_imgs[i] for i in sorted(val_idx)], ROOT/"llvip_ir_val.txt")
write_list(test_imgs, ROOT/"llvip_ir_test.txt")

