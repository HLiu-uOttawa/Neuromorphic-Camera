from pathlib import Path
from dv import AedatFile


class Aedat4Reader:
    """
    Lightweight reader / inspector for Prophesee aedat4 files.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file = None

    # -------- context manager --------
    def __enter__(self):
        self._file = AedatFile(str(self.path))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    # -------- basic info --------
    @property
    def streams(self) -> list[str]:
        return list(self._file.names)

    def has_stream(self, name: str) -> bool:
        return name in self._file.names

    # -------- stream access --------
    def events(self):
        if not self.has_stream("events"):
            raise RuntimeError("No 'events' stream in file")
        return self._file["events"]

    def events_numpy(self):
        return self.events().numpy()

    def imu(self):
        if not self.has_stream("imu"):
            raise RuntimeError("No 'imu' stream in file")
        return self._file["imu"]

    def frames(self):
        if not self.has_stream("frames"):
            raise RuntimeError("No 'frames' stream in file")
        return self._file["frames"]

    # -------- inspection --------
    def inspect(self):
        print(f"File: {self.path.name}")
        print("Streams:")
        for name in self.streams:
            s = self._file[name]
            print(f"  - {name}")
            if hasattr(s, "numpy"):
                arr = s.numpy()
                print(f"      dtype : {arr.dtype}")
                print(f"      count : {len(arr)}")


# ===== factory function =====
def open_aedat4(path: str | Path) -> Aedat4Reader:
    """
    Factory function for aedat4 files.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() != ".aedat4":
        raise ValueError(f"Not an aedat4 file: {path}")

    # quick sanity check
    f = AedatFile(str(path))
    streams = list(f.names)
    f.close()

    if "events" not in streams:
        raise RuntimeError("aedat4 file has no 'events' stream")

    return Aedat4Reader(path)


# ===== self test =====
def _self_test():
    import sys
    import itertools

    if len(sys.argv) < 2:
        print("Usage: python Aedat4Reader.py <file.aedat4> [--full]")
        sys.exit(1)

    path = sys.argv[1]
    full = "--full" in sys.argv[2:]

    print("=== AEDAT4 SELF TEST ===")
    print(f"File: {path}")

    with open_aedat4(path) as r:
        print("[OK] File opened")

        print("Streams:")
        for s in r.streams:
            print(f"  - {s}")

        # events sanity check
        if r.has_stream("events"):
            print("[OK] events stream")
            sample = list(itertools.islice(r.events(), 1000))
            print(f"  sample_count : {len(sample)}")
            if sample:
                e0 = sample[0]
                print(f"  first_event  : t={int(e0.timestamp)} x={int(e0.x)} y={int(e0.y)} p={int(e0.polarity)}")

            if full:
                try:
                    import numpy as np
                    ev_np = np.asarray(r.events().numpy())
                    print(f"  full_count   : {ev_np.shape[0]}")
                    print(f"  dtype        : {ev_np.dtype}")
                except Exception:
                    # fallback: safe counting by iteration
                    full_count = sum(1 for _ in r.events())
                    print(f"  full_count   : {full_count}")


        if r.has_stream("imu"):
            print("[OK] imu stream")

        if r.has_stream("frames"):
            print("[OK] frames stream")

        if r.has_stream("triggers"):
            print("[OK] triggers stream")

    print("=== SELF TEST PASSED ===")



# python Aedat4Reader.py .\ottawa_8.aedat4
# python Aedat4Reader.py .\ottawa_8.aedat4 --full
if __name__ == "__main__":
    _self_test()



