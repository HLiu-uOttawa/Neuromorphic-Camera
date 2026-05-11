from __future__ import annotations

import os
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
from dv import AedatFile

from metavision_core.event_io import DatWriter


def _ensure_unique_path(path: str) -> str:
    """If file exists, append _1/_2/... before extension."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def convert_aedat4_to_prophesee_hdf5(
    aedat_path: str,
    out_h5: str | None = None,
    width: int = 346,
    height: int = 260,
    tmp_dat: str | None = None,
    ts_unit: str = "us",            # input timestamp unit from dv: us/ns/ms
    keep_tmp_dat: bool = False,
    flush_events: int = 1_000_000,  # buffering for single-event mode
    zero_start: bool = True,        # rebase timestamps so first event starts at 0
    force_unique_out: bool = True,  # auto rename output if exists/locked
):
    """
    AEDAT4 -> DAT (CD events) -> Prophesee HDF5 event file (playable in Metavision tools).

    Fixes:
      - tqdm never calls len() on dv streams
      - supports dv packet(EventStore) AND single-event iteration
      - optional timestamp rebase to start at 0 (default True)
      - auto-rename output if exists/locked (Windows)
    """
    aedat_path = os.path.abspath(aedat_path)
    if not os.path.exists(aedat_path):
        raise FileNotFoundError(f"Input AEDAT4 file not found: {aedat_path}")

    if out_h5 is None:
        base, _ = os.path.splitext(aedat_path)
        out_h5 = base + ".prophesee.hdf5"
    out_h5 = os.path.abspath(out_h5)
    if force_unique_out:
        out_h5 = _ensure_unique_path(out_h5)

    if tmp_dat is None:
        base, _ = os.path.splitext(out_h5)
        tmp_dat = base + ".tmp.dat"
    tmp_dat = os.path.abspath(tmp_dat)

    tool = shutil.which("metavision_file_to_hdf5")
    if tool is None:
        raise RuntimeError(
            "Cannot find 'metavision_file_to_hdf5' in PATH.\n"
            "Please ensure Prophesee bin directory is in PATH."
        )

    # Timestamp scaling to microseconds (DAT expects us)
    if ts_unit == "us":
        scale = 1.0
    elif ts_unit == "ns":
        scale = 1.0 / 1000.0
    elif ts_unit == "ms":
        scale = 1000.0
    else:
        raise ValueError("ts_unit must be one of: us, ns, ms")

    print(f"[INFO] Input AEDAT4: {aedat_path}")
    print(f"[INFO] Output HDF5 : {out_h5}")
    print(f"[INFO] Temp DAT   : {tmp_dat}")
    print(f"[INFO] Geometry  : {width}x{height}")
    print(f"[INFO] ts_unit   : {ts_unit} -> stored as microseconds")
    print(f"[INFO] zero_start: {zero_start}")

    # -----------------------
    # 1) AEDAT4 -> DAT
    # -----------------------
    with AedatFile(aedat_path) as f:
        if "events" not in f.names:
            raise RuntimeError("No 'events' stream found in this AEDAT4 file.")

        writer = DatWriter(tmp_dat, height=int(height), width=int(width))

        total = 0
        last_t_written = None

        # for diagnostics (raw timestamps in us)
        raw_t0 = None
        raw_t_last = None

        # for rebase (us)
        t0_us = None

        events_iter = iter(f["events"])
        try:
            first = next(events_iter)
        except StopIteration:
            writer.close()
            raise RuntimeError("Event stream is empty.")

        packet_mode = hasattr(first, "numpy")
        pbar = tqdm(desc="AEDAT4 -> DAT", unit="evt")

        def rebase_if_needed(t_us_arr: np.ndarray) -> np.ndarray:
            nonlocal t0_us
            if not zero_start:
                return t_us_arr
            if t0_us is None:
                t0_us = int(t_us_arr[0])
            return t_us_arr - t0_us

        def write_batch(x, y, p01, t_us_raw):
            """t_us_raw must be int64 microseconds (NOT rebased yet)."""
            nonlocal total, last_t_written, raw_t0, raw_t_last

            if t_us_raw.size == 0:
                return

            if raw_t0 is None:
                raw_t0 = int(t_us_raw[0])
            raw_t_last = int(t_us_raw[-1])

            t_us = rebase_if_needed(t_us_raw)

            # DatWriter requires chronological order across writes
            if last_t_written is not None and t_us[0] < last_t_written:
                # If this happens, data is out-of-order; sort within batch (safe)
                order = np.argsort(t_us, kind="mergesort")
                t_us = t_us[order]
                x = x[order]
                y = y[order]
                p01 = p01[order]

                if t_us[0] < last_t_written:
                    raise RuntimeError(
                        f"Non-monotonic timestamps across batches even after sorting. "
                        f"batch_first={t_us[0]}, last_written={last_t_written}"
                    )

            ev = np.empty(t_us.shape[0], dtype=[("x", np.uint16), ("y", np.uint16), ("p", np.int16), ("t", np.int64)])
            ev["x"] = x.astype(np.uint16, copy=False)
            ev["y"] = y.astype(np.uint16, copy=False)
            ev["p"] = p01.astype(np.int16, copy=False)
            ev["t"] = t_us.astype(np.int64, copy=False)

            writer.write(ev)
            last_t_written = int(ev["t"][-1])

            total += ev.shape[0]
            pbar.update(ev.shape[0])

        if packet_mode:
            def handle_packet(pkt):
                arr = pkt.numpy()
                if arr.size == 0:
                    return
                names = arr.dtype.names
                if names is None:
                    raise RuntimeError("Unexpected packet dtype (no named fields).")

                t_in = arr["timestamp"] if "timestamp" in names else arr["t"]
                x = arr["x"]
                y = arr["y"]
                p_in = arr["polarity"] if "polarity" in names else arr["p"]

                t_us_raw = np.asarray(np.round(np.asarray(t_in, dtype=np.float64) * scale), dtype=np.int64)
                p01 = (np.asarray(p_in) > 0).astype(np.int16)

                write_batch(x, y, p01, t_us_raw)

            handle_packet(first)
            for pkt in events_iter:
                handle_packet(pkt)

        else:
            # single-event mode: buffer RAW(us) timestamps, do rebase ONLY in write_batch
            buf_x, buf_y, buf_p, buf_t = [], [], [], []

            def flush():
                nonlocal buf_x, buf_y, buf_p, buf_t
                if not buf_t:
                    return
                x = np.asarray(buf_x, dtype=np.uint16)
                y = np.asarray(buf_y, dtype=np.uint16)
                p01 = np.asarray(buf_p, dtype=np.int16)
                t_us_raw = np.asarray(buf_t, dtype=np.int64)  # RAW us
                write_batch(x, y, p01, t_us_raw)
                buf_x, buf_y, buf_p, buf_t = [], [], [], []

            def handle_event(evt):
                x = int(getattr(evt, "x"))
                y = int(getattr(evt, "y"))

                t_in = getattr(evt, "timestamp", None)
                if t_in is None:
                    t_in = getattr(evt, "t")

                p_in = getattr(evt, "polarity", None)
                if p_in is None:
                    p_in = getattr(evt, "p")

                t_us_raw = int(round(float(t_in) * scale))  # RAW us
                p01 = 1 if int(p_in) > 0 else 0

                buf_x.append(x)
                buf_y.append(y)
                buf_t.append(t_us_raw)
                buf_p.append(p01)

                if len(buf_t) >= int(flush_events):
                    flush()

            handle_event(first)
            for evt in events_iter:
                handle_event(evt)
            flush()

        pbar.close()
        writer.close()

    if raw_t0 is not None and raw_t_last is not None:
        dur_us = raw_t_last - raw_t0
        print(f"[INFO] raw t0(us):   {raw_t0}")
        print(f"[INFO] raw t_last:   {raw_t_last}")
        print(f"[INFO] raw duration: {dur_us/1e6:.3f} s  ({dur_us} us)")

    print(f"[OK] DAT written: {tmp_dat}")
    print(f"     events: {total:,}")

    # -----------------------
    # 2) DAT -> HDF5
    # -----------------------
    cmd = [tool, "-i", tmp_dat, "-o", out_h5]
    print("[RUN]", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        # Common Windows lock case: retry with a new name once
        if force_unique_out:
            out_h5_2 = _ensure_unique_path(out_h5)
            if out_h5_2 != out_h5:
                print(f"[WARN] HDF5 write failed. Retrying with new output name:\n       {out_h5_2}")
                cmd2 = [tool, "-i", tmp_dat, "-o", out_h5_2]
                print("[RUN]", " ".join(cmd2))
                subprocess.check_call(cmd2)
                out_h5 = out_h5_2
            else:
                raise
        else:
            raise

    print(f"[OK] Prophesee playable HDF5 written: {out_h5}")

    if not keep_tmp_dat:
        try:
            os.remove(tmp_dat)
            print(f"[CLEAN] removed tmp dat: {tmp_dat}")
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AEDAT4 -> DAT -> Prophesee playable HDF5 (with timestamp rebase)")
    ap.add_argument("aedat4", help="Input .aedat4 file path")
    ap.add_argument("--out", default=None, help="Output .hdf5 path")
    ap.add_argument("--width", type=int, default=346, help="Sensor width")
    ap.add_argument("--height", type=int, default=260, help="Sensor height")
    ap.add_argument("--ts-unit", default="us", choices=["us", "ns", "ms"], help="AEDAT4 timestamp unit")
    ap.add_argument("--keep-dat", action="store_true", help="Keep intermediate .dat file")
    ap.add_argument("--flush-events", type=int, default=1_000_000, help="Buffer size for single-event mode")
    ap.add_argument("--zero-start", action="store_true", help="Rebase timestamps so the first event starts at 0 (recommended)")
    ap.add_argument("--no-zero-start", action="store_true", help="Do NOT rebase timestamps")
    ap.add_argument("--no-unique-out", action="store_true", help="Do NOT auto-rename output file if it already exists/locked")
    args = ap.parse_args()

    zero_start = True
    if args.no_zero_start:
        zero_start = False
    if args.zero_start:
        zero_start = True

    convert_aedat4_to_prophesee_hdf5(
        args.aedat4,
        out_h5=args.out,
        width=args.width,
        height=args.height,
        ts_unit=args.ts_unit,
        keep_tmp_dat=args.keep_dat,
        flush_events=args.flush_events,
        zero_start=zero_start,
        force_unique_out=(not args.no_unique_out),
    )
