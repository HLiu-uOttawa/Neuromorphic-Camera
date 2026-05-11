# aedat4_to_prophesee_hdf5.py
from __future__ import annotations

import os
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
from dv import AedatFile

# Prophesee Python API (Metavision SDK)
from metavision_core.event_io import DatWriter


def aedat4_to_prophesee_hdf5(
    aedat_path: str,
    out_h5_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    tmp_dat_path: str | None = None,
    assume_ts_unit: str = "us",
):
    """
    Convert AEDAT4 to Prophesee-playable HDF5 by:
      1) AEDAT4 -> DAT (with DatWriter)
      2) DAT -> HDF5 event file (metavision_file_to_hdf5)

    Notes:
      - Prophesee HDF5 event files use ECF codec and a specific structure;
        Python does NOT provide an HDF5 event writer, so we rely on the official converter tool.
      - Timestamps in Metavision event files are in microseconds (t in us) for CD events.

    Args:
      aedat_path: input .aedat4
      out_h5_path: output .hdf5 (default: same base name)
      width/height: sensor geometry. If None, try to infer from AEDAT4 header; else required.
      tmp_dat_path: intermediate .dat path. If None, create next to output.
      assume_ts_unit: "us" or "ns" or "ms" – used to scale dv timestamps to microseconds if needed.
                      Many dv streams are already in microseconds; if yours is different, set it here.
    """
    aedat_path = os.path.abspath(aedat_path)
    if out_h5_path is None:
        base, _ = os.path.splitext(aedat_path)
        out_h5_path = base + ".hdf5"
    out_h5_path = os.path.abspath(out_h5_path)

    if tmp_dat_path is None:
        base, _ = os.path.splitext(out_h5_path)
        tmp_dat_path = base + ".tmp.dat"
    tmp_dat_path = os.path.abspath(tmp_dat_path)

    # check converter tool exists
    tool = shutil.which("metavision_file_to_hdf5")
    if tool is None:
        raise RuntimeError(
            "Cannot find 'metavision_file_to_hdf5' in PATH. "
            "Please install Metavision SDK and ensure its bin folder is in PATH."
        )

    # time scaling to microseconds for Metavision
    if assume_ts_unit == "us":
        scale_to_us = 1.0
    elif assume_ts_unit == "ns":
        scale_to_us = 1.0 / 1000.0
    elif assume_ts_unit == "ms":
        scale_to_us = 1000.0
    else:
        raise ValueError("assume_ts_unit must be one of: 'us', 'ns', 'ms'")

    with AedatFile(aedat_path) as f:
        if "events" not in f.names:
            raise RuntimeError("No 'events' stream found in this AEDAT4 file.")

        # Try to infer width/height if not provided
        if width is None or height is None:
            # Best-effort: dv header may contain geometry in some versions
            # If it fails, you must pass width/height explicitly.
            w = h = None
            try:
                if hasattr(f, "header") and f.header is not None:
                    hdr = dict(f.header)
                    # common keys vary; try a few patterns
                    for k in ["width", "W", "sensor_width"]:
                        if k in hdr:
                            w = int(hdr[k])
                            break
                    for k in ["height", "H", "sensor_height"]:
                        if k in hdr:
                            h = int(hdr[k])
                            break
            except Exception:
                pass

            if w is None or h is None:
                raise RuntimeError(
                    "Cannot infer sensor width/height from AEDAT4 header. "
                    "Please re-run with --width and --height."
                )
            width, height = w, h

        # Write DAT (decoded events)
        writer = DatWriter(tmp_dat_path, height=int(height), width=int(width))

        total = 0
        for pkt in tqdm(f["events"], desc="AEDAT4 -> DAT", unit="pkt"):
            # Convert packet to numpy structured array
            arr = pkt.numpy() if hasattr(pkt, "numpy") else np.array(list(pkt))
            if arr.size == 0:
                continue

            names = arr.dtype.names
            if names is None:
                raise RuntimeError("Unexpected AEDAT4 packet dtype (no named fields).")

            # Map fields
            t = arr["timestamp"] if "timestamp" in names else arr["t"]
            x = arr["x"]
            y = arr["y"]
            p = arr["polarity"] if "polarity" in names else arr["p"]

            # Scale timestamps to microseconds (int64)
            t_us = np.asarray(np.round(np.asarray(t, dtype=np.float64) * scale_to_us), dtype=np.int64)

            # Metavision expects polarity as int16 in many internal structures; DatWriter handles Event2D,
            # but we'll convert to {0,1} int16 to be safe.
            p01 = (np.asarray(p) > 0).astype(np.int16)

            # Build Event2D structured array expected by DatWriter:
            # fields: x (uint16), y (uint16), p (int16), t (int64)   (t in us)
            ev = np.empty(t_us.shape[0], dtype=[("x", np.uint16), ("y", np.uint16), ("p", np.int16), ("t", np.int64)])
            ev["x"] = x.astype(np.uint16, copy=False)
            ev["y"] = y.astype(np.uint16, copy=False)
            ev["p"] = p01
            ev["t"] = t_us

            writer.write(ev)
            total += ev.shape[0]

        writer.close()
        print(f"[OK] Wrote DAT: {tmp_dat_path}  (events={total:,})")

    # Convert DAT -> Prophesee HDF5 event file (ECF + proper layout)
    cmd = [tool, "-i", tmp_dat_path, "-o", out_h5_path]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

    print(f"[OK] Wrote Prophesee HDF5 event file: {out_h5_path}")
    print("You should be able to open it in Metavision Studio/Viewer.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert AEDAT4 to Prophesee-playable HDF5 event file.")
    ap.add_argument("aedat4", help="Input .aedat4 path")
    ap.add_argument("--out", default=None, help="Output .hdf5 path")
    ap.add_argument("--width", type=int, default=None, help="Sensor width (pixels)")
    ap.add_argument("--height", type=int, default=None, help="Sensor height (pixels)")
    ap.add_argument("--tmp-dat", default=None, help="Intermediate .dat path")
    ap.add_argument("--ts-unit", default="us", choices=["us", "ns", "ms"], help="AEDAT4 timestamp unit (for scaling to us)")
    args = ap.parse_args()

    aedat4_to_prophesee_hdf5(
        args.aedat4,
        out_h5_path=args.out,
        width=args.width,
        height=args.height,
        tmp_dat_path=args.tmp_dat,
        assume_ts_unit=args.ts_unit,
    )
