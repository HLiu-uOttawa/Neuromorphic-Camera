import zipfile
from pathlib import Path

# Config
RAW_ZIP_DIR = Path("fred_raw_zips")  # original zip folder
OUT_TRAIN = Path("train")
OUT_TEST  = Path("test")
OUT_VAL   = Path("val")

TRAIN_RANGE = range(0, 100)     # 0 ~ 99
TEST_RANGE  = range(100, 200)   # 100 ~ 199
# 

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unzip_to(zip_path: Path, out_dir: Path):
    seq_name = zip_path.stem          # e.g. "0", "123"
    target = out_dir / seq_name       # train/0 , test/123
    ensure_dir(target)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target)

    print(f"[OK] {zip_path.name} -> {target}")

def main():
    if not RAW_ZIP_DIR.exists():
        raise FileNotFoundError(f"Not found: {RAW_ZIP_DIR}")

    # Create train / test / val
    ensure_dir(OUT_TRAIN)
    ensure_dir(OUT_TEST)
    ensure_dir(OUT_VAL)

    zip_files = sorted(RAW_ZIP_DIR.glob("*.zip"), key=lambda p: int(p.stem))

    for zip_path in zip_files:
        try:
            idx = int(zip_path.stem)
        except ValueError:
            print(f"[SKIP] not a numeric zip: {zip_path.name}")
            continue

        if idx in TRAIN_RANGE:
            unzip_to(zip_path, OUT_TRAIN)
        elif idx in TEST_RANGE:
            unzip_to(zip_path, OUT_TEST)
        else:
            print(f"[SKIP] {zip_path.name} (not in train/test range)")

    print("\nDONE.")
    print(f"Train dir: {OUT_TRAIN.resolve()}")
    print(f"Test  dir: {OUT_TEST.resolve()}")
    print(f"Val   dir: {OUT_VAL.resolve()} (empty for now)")

if __name__ == "__main__":
    main()
