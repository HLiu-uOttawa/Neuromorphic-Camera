from pathlib import Path
import shutil
import csv

FRED_ROOT = Path("./FRED/fred_raw_unzip/")

OUT_RGB = Path("./FRED/fred_for_yolo/fred_yolo_rgb")
OUT_EVT = Path("./FRED/fred_for_yolo/FRED_yolo_event")

RGB_IMG_DIRNAME = "RGB"                 # RGB folder: .../RGB/*.jpg
EVT_IMG_DIRNAME = Path("Event/Frames")  # Event folder: .../Event/Frames/*.png

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_out(out_root: Path):
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def list_images(img_dir: Path):
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda x: x.name)  # Sort by name to ensure a stable order
    return imgs


def list_labels(lbl_dir: Path):
    labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    labels.sort(key=lambda x: x.name)
    return labels


def write_yaml(out_root: Path, class_names):
    yaml_path = out_root / "data.yaml"
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, n in enumerate(class_names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {yaml_path}")


def pack_modality(modality: str, out_root: Path, img_rel: Path, lbl_dirname: str):
    """
    modality: "RGB" or "Event"
    img_rel:  RGB -> Path("RGB")
              Event -> Path("Event/Frames")
    lbl_dirname: "RGB_YOLO" or "Event_YOLO"  (仅 labels)
    """
    ensure_out(out_root)
    log_path = out_root / "pairing_log.csv"
    log_rows = []

    for split in ["train", "val", "test"]:
        split_dir = FRED_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] Missing split: {split_dir}")
            continue

        seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda x: x.name)

        total_pairs = 0
        total_skipped = 0

        for seq in seq_dirs:
            img_dir = seq / img_rel
            lbl_dir = seq / lbl_dirname

            if not img_dir.exists() or not lbl_dir.exists():
                continue

            imgs = list_images(img_dir)
            labels = list_labels(lbl_dir)

            if len(imgs) != len(labels):
                print(f"[WARN] count mismatch: {split}/{seq.name} {modality} imgs={len(imgs)} labels={len(labels)} -> SKIP this seq")
                total_skipped += 1
                continue

            for i, (img_path, lbl_path) in enumerate(zip(imgs, labels), start=1):
                # Unify output filenames: use the image name as the primary key
                # To avoid filename conflicts across different sequences, add the sequence prefix
                stem = f"{seq.name}_{img_path.stem}"
                dst_img = out_root / "images" / split / f"{stem}{img_path.suffix.lower()}"
                dst_lbl = out_root / "labels" / split / f"{stem}.txt"

                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

                log_rows.append({
                    "split": split,
                    "seq": seq.name,
                    "index": i,
                    "src_image": str(img_path),
                    "src_label": str(lbl_path),
                    "dst_image": str(dst_img),
                    "dst_label": str(dst_lbl),
                })
                total_pairs += 1

        print(f"[OK] {modality} split={split}: paired_samples={total_pairs}, skipped_seqs={total_skipped}")

    # Write pairing logs for easy spot-checking
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else
                           ["split","seq","index","src_image","src_label","dst_image","dst_label"])
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    print(f"[OK] wrote pairing log: {log_path}")


if __name__ == "__main__":
    # If only have one class, “drone detection,” this is sufficient
    CLASS_NAMES = ["drone"]

    # RGB: images in .../RGB/, labels in .../RGB_YOLO/
    pack_modality(
        modality="RGB",
        out_root=OUT_RGB,
        img_rel=Path(RGB_IMG_DIRNAME),
        lbl_dirname="RGB_YOLO"
    )
    write_yaml(OUT_RGB, CLASS_NAMES)

    # Event: images in .../Event/Frames/, labels in .../Event_YOLO/
    pack_modality(
        modality="Event",
        out_root=OUT_EVT,
        img_rel=EVT_IMG_DIRNAME,
        lbl_dirname="Event_YOLO"
    )
    write_yaml(OUT_EVT, CLASS_NAMES)

    print("Done.")
