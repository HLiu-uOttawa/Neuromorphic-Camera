from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterable, Any, Dict, Tuple
import itertools
import numpy as np
from dv import AedatFile


@dataclass
class StreamSummary:
    name: str
    sample_count: int
    first_item: Dict[str, Any]


@dataclass
class AedatSummary:
    file: Path
    streams: list[str]
    stream_summaries: list[StreamSummary]
    time_ns_first: Optional[int]
    time_ns_last_est: Optional[int]


class Aedat4Reader:
    """Lightweight reader for Prophesee aedat4 files (dv)."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file: Optional[AedatFile] = None

    def __enter__(self) -> "Aedat4Reader":
        self._file = AedatFile(str(self.path))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def streams(self) -> list[str]:
        return list(self._file.names)

    def has_stream(self, name: str) -> bool:
        return name in self._file.names

    def stream(self, name: str):
        if not self.has_stream(name):
            raise RuntimeError(f"No '{name}' stream in file")
        return self._file[name]

    # convenience
    def events(self): return self.stream("events")
    def frames(self): return self.stream("frames")
    def imu(self): return self.stream("imu")
    def triggers(self): return self.stream("triggers")


def open_aedat4(path: str | Path) -> Aedat4Reader:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() != ".aedat4":
        raise ValueError(f"Not an aedat4 file: {p}")
    # quick sanity: must have events
    f = AedatFile(str(p))
    names = list(f.names)
    f.close()
    if "events" not in names:
        raise RuntimeError("aedat4 file has no 'events' stream")
    return Aedat4Reader(p)


def _first_event_dict(e) -> Dict[str, Any]:
    # dv event object uses attributes
    return {
        "timestamp": int(getattr(e, "timestamp")),
        "x": int(getattr(e, "x")),
        "y": int(getattr(e, "y")),
        "polarity": int(getattr(e, "polarity")),
    }


def _first_frame_dict(fr) -> Dict[str, Any]:
    # dv frame objects usually have timestamp + image (may be .image)
    d: Dict[str, Any] = {}
    if hasattr(fr, "timestamp"):
        d["timestamp"] = int(fr.timestamp)
    # try get image shape
    img = None
    for attr in ("image", "frame", "data"):
        if hasattr(fr, attr):
            img = getattr(fr, attr)
            break
    if img is not None:
        try:
            arr = np.asarray(img)
            d["image_shape"] = tuple(arr.shape)
            d["image_dtype"] = str(arr.dtype)
        except Exception:
            d["image_shape"] = "unknown"
    return d


def _first_imu_dict(im) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if hasattr(im, "timestamp"):
        d["timestamp"] = int(im.timestamp)
    # common fields (may vary by device)
    for k in ("acceleration", "accelerometer", "accel", "gyro", "gyroscope", "angular_velocity", "temperature"):
        if hasattr(im, k):
            v = getattr(im, k)
            try:
                d[k] = np.asarray(v).astype(float).tolist()
            except Exception:
                d[k] = str(v)
    return d


def _first_trigger_dict(tr) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if hasattr(tr, "timestamp"):
        d["timestamp"] = int(tr.timestamp)
    for k in ("id", "value", "polarity", "source"):
        if hasattr(tr, k):
            try:
                d[k] = int(getattr(tr, k))
            except Exception:
                d[k] = str(getattr(tr, k))
    return d


def summarize_aedat4(path: str | Path, sample_n: int = 1000) -> AedatSummary:
    p = Path(path).expanduser().resolve()
    stream_summaries: list[StreamSummary] = []
    t0: Optional[int] = None
    t_last_est: Optional[int] = None

    with open_aedat4(p) as r:
        streams = r.streams

        # events: sample first and also estimate last timestamp by sampling tail-ish is not supported;
        # we do a lightweight pass over events only for last timestamp if sample_n is small.
        if r.has_stream("events"):
            it = iter(r.events())
            sample = list(itertools.islice(it, sample_n))
            if sample:
                t0 = int(sample[0].timestamp)
                # estimate last timestamp: keep updating from sampled iterator only (not full pass)
                t_last_est = int(sample[-1].timestamp)
                stream_summaries.append(StreamSummary("events", len(sample), _first_event_dict(sample[0])))

        if r.has_stream("frames"):
            it = iter(r.frames())
            sample = list(itertools.islice(it, min(10, sample_n)))
            if sample:
                stream_summaries.append(StreamSummary("frames", len(sample), _first_frame_dict(sample[0])))

        if r.has_stream("imu"):
            it = iter(r.imu())
            sample = list(itertools.islice(it, min(50, sample_n)))
            if sample:
                stream_summaries.append(StreamSummary("imu", len(sample), _first_imu_dict(sample[0])))

        if r.has_stream("triggers"):
            it = iter(r.triggers())
            sample = list(itertools.islice(it, min(50, sample_n)))
            if sample:
                stream_summaries.append(StreamSummary("triggers", len(sample), _first_trigger_dict(sample[0])))

    return AedatSummary(
        file=p,
        streams=streams,
        stream_summaries=stream_summaries,
        time_ns_first=t0,
        time_ns_last_est=t_last_est,
    )


def print_summary(s: AedatSummary) -> None:
    print("=== AEDAT4 SUMMARY ===")
    print(f"File: {s.file}")
    print(f"Streams: {s.streams}")
    if s.time_ns_first is not None:
        print(f"First timestamp (ns): {s.time_ns_first}")
    if s.time_ns_last_est is not None:
        print(f"Last timestamp est (ns, from sample): {s.time_ns_last_est}")
    for ss in s.stream_summaries:
        print(f"\n[{ss.name}] sample_count={ss.sample_count}")
        for k, v in ss.first_item.items():
            print(f"  {k}: {v}")

# python -c "from aedat_tools import summarize_aedat4, print_summary; s=summarize_aedat4(r'.\ottawa_8.aedat4'); print_summary(s)"
# 
import h5py

def _ensure_group(h5: h5py.File, path: str):
    if path in h5:
        return h5[path]
    return h5.create_group(path)

def _append_1d(ds, data: np.ndarray):
    n0 = ds.shape[0]
    n1 = n0 + data.shape[0]
    ds.resize((n1,))
    ds[n0:n1] = data

def _append_2d(ds, data: np.ndarray):
    n0 = ds.shape[0]
    n1 = n0 + data.shape[0]
    ds.resize((n1, ds.shape[1]))
    ds[n0:n1, :] = data

def export_hdf5(aedat_path: str | Path, h5_path: str | Path,
                chunk_events: int = 2_000_000,
                chunk_imu: int = 100_000,
                chunk_trig: int = 100_000,
                chunk_frames: int = 2000,
                compression: str = "gzip") -> Path:
    aedat_path = Path(aedat_path).expanduser().resolve()
    h5_path = Path(h5_path).expanduser().resolve()

    with open_aedat4(aedat_path) as r:
        # t0 from first event
        first_ev = next(iter(r.events()))
        t0_ns = int(first_ev.timestamp)

    with h5py.File(h5_path, "w") as h5, open_aedat4(aedat_path) as r:
        # root attrs
        h5.attrs["format"] = "aedat4_to_hdf5_v1"
        h5.attrs["source_file"] = aedat_path.name
        h5.attrs["time_unit"] = "ns"
        h5.attrs["time_domain"] = "monotonic_unknown_origin"
        h5.attrs["t0_ns"] = t0_ns

        # ---------- events ----------
        if r.has_stream("events"):
            g = _ensure_group(h5, "events")
            g.create_dataset("t_ns_raw", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_events, 200000),), compression=compression)
            g.create_dataset("t_ns_rel", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_events, 200000),), compression=compression)
            g.create_dataset("x", shape=(0,), maxshape=(None,),
                             dtype=np.uint16, chunks=(min(chunk_events, 200000),), compression=compression)
            g.create_dataset("y", shape=(0,), maxshape=(None,),
                             dtype=np.uint16, chunks=(min(chunk_events, 200000),), compression=compression)
            g.create_dataset("p", shape=(0,), maxshape=(None,),
                             dtype=np.uint8, chunks=(min(chunk_events, 200000),), compression=compression)

            buf_t = np.empty((0,), np.int64)
            buf_x = np.empty((0,), np.uint16)
            buf_y = np.empty((0,), np.uint16)
            buf_p = np.empty((0,), np.uint8)

            # stream iterate
            t_list, x_list, y_list, p_list = [], [], [], []
            for e in r.events():
                t_list.append(int(e.timestamp))
                x_list.append(int(e.x))
                y_list.append(int(e.y))
                p_list.append(int(e.polarity))

                if len(t_list) >= 200000:  # flush micro-batch to numpy
                    t_arr = np.asarray(t_list, dtype=np.int64)
                    x_arr = np.asarray(x_list, dtype=np.uint16)
                    y_arr = np.asarray(y_list, dtype=np.uint16)
                    p_arr = np.asarray(p_list, dtype=np.uint8)

                    _append_1d(g["t_ns_raw"], t_arr)
                    _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                    _append_1d(g["x"], x_arr)
                    _append_1d(g["y"], y_arr)
                    _append_1d(g["p"], p_arr)

                    t_list.clear(); x_list.clear(); y_list.clear(); p_list.clear()

            # final flush
            if t_list:
                t_arr = np.asarray(t_list, dtype=np.int64)
                x_arr = np.asarray(x_list, dtype=np.uint16)
                y_arr = np.asarray(y_list, dtype=np.uint16)
                p_arr = np.asarray(p_list, dtype=np.uint8)
                _append_1d(g["t_ns_raw"], t_arr)
                _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                _append_1d(g["x"], x_arr)
                _append_1d(g["y"], y_arr)
                _append_1d(g["p"], p_arr)

        # ---------- frames ----------
        if r.has_stream("frames"):
            g = _ensure_group(h5, "frames")

            # get first frame to determine shape
            fr_it = iter(r.frames())
            first_fr = next(fr_it, None)
            if first_fr is not None:
                # image array
                img = None
                for attr in ("image", "frame", "data"):
                    if hasattr(first_fr, attr):
                        img = getattr(first_fr, attr)
                        break
                img_arr = np.asarray(img)
                frame_shape = img_arr.shape  # (H,W) or (H,W,C)

                g.create_dataset("t_ns_raw", shape=(0,), maxshape=(None,),
                                 dtype=np.int64, chunks=(min(chunk_frames, 2000),), compression=compression)
                g.create_dataset("t_ns_rel", shape=(0,), maxshape=(None,),
                                 dtype=np.int64, chunks=(min(chunk_frames, 2000),), compression=compression)

                # image dataset (K, ...)
                g.create_dataset("image", shape=(0, *frame_shape), maxshape=(None, *frame_shape),
                                 dtype=img_arr.dtype, chunks=(1, *frame_shape), compression=compression)

                # write first + rest
                def append_frame(ts: int, image_np: np.ndarray):
                    n0 = g["image"].shape[0]
                    g["image"].resize((n0 + 1, *frame_shape))
                    g["image"][n0] = image_np
                    _append_1d(g["t_ns_raw"], np.asarray([ts], np.int64))
                    _append_1d(g["t_ns_rel"], np.asarray([ts - t0_ns], np.int64))

                append_frame(int(first_fr.timestamp), img_arr)

                for fr in fr_it:
                    img = None
                    for attr in ("image", "frame", "data"):
                        if hasattr(fr, attr):
                            img = getattr(fr, attr)
                            break
                    append_frame(int(fr.timestamp), np.asarray(img))

        # ---------- imu ----------
        if r.has_stream("imu"):
            g = _ensure_group(h5, "imu")
            g.create_dataset("t_ns_raw", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_imu, 200000),), compression=compression)
            g.create_dataset("t_ns_rel", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_imu, 200000),), compression=compression)
            g.create_dataset("accel", shape=(0,3), maxshape=(None,3),
                             dtype=np.float32, chunks=(min(chunk_imu, 200000),3), compression=compression)
            g.create_dataset("gyro", shape=(0,3), maxshape=(None,3),
                             dtype=np.float32, chunks=(min(chunk_imu, 200000),3), compression=compression)

            t_list, a_list, g_list = [], [], []
            for im in r.imu():
                t_list.append(int(im.timestamp))

                # try common attribute names
                acc = None
                for k in ("acceleration", "accelerometer", "accel"):
                    if hasattr(im, k):
                        acc = getattr(im, k); break
                gyr = None
                for k in ("gyroscope", "gyro", "angular_velocity"):
                    if hasattr(im, k):
                        gyr = getattr(im, k); break

                acc = np.asarray(acc, dtype=np.float32).reshape(3,)
                gyr = np.asarray(gyr, dtype=np.float32).reshape(3,)
                a_list.append(acc)
                g_list.append(gyr)

                if len(t_list) >= 50000:
                    t_arr = np.asarray(t_list, np.int64)
                    a_arr = np.asarray(a_list, np.float32)
                    gg_arr = np.asarray(g_list, np.float32)
                    _append_1d(g["t_ns_raw"], t_arr)
                    _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                    _append_2d(g["accel"], a_arr)
                    _append_2d(g["gyro"], gg_arr)
                    t_list.clear(); a_list.clear(); g_list.clear()

            if t_list:
                t_arr = np.asarray(t_list, np.int64)
                a_arr = np.asarray(a_list, np.float32)
                gg_arr = np.asarray(g_list, np.float32)
                _append_1d(g["t_ns_raw"], t_arr)
                _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                _append_2d(g["accel"], a_arr)
                _append_2d(g["gyro"], gg_arr)

        # ---------- triggers ----------
        if r.has_stream("triggers"):
            g = _ensure_group(h5, "triggers")
            g.create_dataset("t_ns_raw", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_trig, 200000),), compression=compression)
            g.create_dataset("t_ns_rel", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(min(chunk_trig, 200000),), compression=compression)
            g.create_dataset("id", shape=(0,), maxshape=(None,),
                             dtype=np.int16, chunks=(min(chunk_trig, 200000),), compression=compression)
            g.create_dataset("value", shape=(0,), maxshape=(None,),
                             dtype=np.int16, chunks=(min(chunk_trig, 200000),), compression=compression)

            t_list, id_list, v_list = [], [], []
            for tr in r.triggers():
                t_list.append(int(getattr(tr, "timestamp")))
                tid = int(getattr(tr, "id", 0))
                val = int(getattr(tr, "value", getattr(tr, "polarity", 0)))
                id_list.append(tid)
                v_list.append(val)

                if len(t_list) >= 50000:
                    t_arr = np.asarray(t_list, np.int64)
                    id_arr = np.asarray(id_list, np.int16)
                    v_arr = np.asarray(v_list, np.int16)
                    _append_1d(g["t_ns_raw"], t_arr)
                    _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                    _append_1d(g["id"], id_arr)
                    _append_1d(g["value"], v_arr)
                    t_list.clear(); id_list.clear(); v_list.clear()

            if t_list:
                t_arr = np.asarray(t_list, np.int64)
                id_arr = np.asarray(id_list, np.int16)
                v_arr = np.asarray(v_list, np.int16)
                _append_1d(g["t_ns_raw"], t_arr)
                _append_1d(g["t_ns_rel"], t_arr - t0_ns)
                _append_1d(g["id"], id_arr)
                _append_1d(g["value"], v_arr)

    return h5_path

# python -c "from aedat_tools import export_hdf5; export_hdf5(r'.\ottawa_8.aedat4', r'.\ottawa_8.h5')"
