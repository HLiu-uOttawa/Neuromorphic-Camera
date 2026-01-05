# Aedat4Data.py
import numpy as np
from tqdm import tqdm
from dv import AedatFile

class Aedat4Data:
    def __init__(self, file_path, auto_load=True):
        self.file_path = file_path
        self.aedat = None
        if auto_load:
            self._load()

    def _load(self):
        """Open AEDAT4 file"""
        self.aedat = AedatFile(self.file_path)

    def read_events(self):
        pass

    def read_all_events(self, as_numpy=True, flush_every=1_000_000):
        """
        Read ALL events into memory with a progress bar.
        This version NEVER calls len() on dv iterators.

        Args:
            as_numpy: return structured numpy array if True else list
            flush_every: print occasional progress memory-friendly (optional)

        Returns:
            np.ndarray or list of tuples (t, x, y, p)
        """
        aedat = AedatFile(self.file_path)
        if "events" not in aedat.names:
            return None

        events = []
        print("Reading all events...")

        pbar = tqdm(total=0, unit="ev", unit_scale=True, dynamic_ncols=True)
        count = 0

        try:
            for ev in aedat["events"]:
                events.append((
                    int(ev.timestamp),
                    int(ev.x),
                    int(ev.y),
                    int(ev.polarity),
                ))
                count += 1

                # update tqdm in batches for speed
                if (count % 10000) == 0:
                    pbar.update(10000)

                # (optional) occasional message, can remove
                if flush_every and (count % flush_every) == 0:
                    pbar.set_postfix_str(f"stored={len(events):,}")
        finally:
            # update remaining
            rem = count % 10000
            if rem:
                pbar.update(rem)
            pbar.close()

        if not as_numpy:
            return events

        return np.array(
            events,
            dtype=[("t", "int64"), ("x", "int16"), ("y", "int16"), ("p", "int8")],
        )

    def read_frames(self, max_frames=None):
        """
        Read APS frame stream.

        Returns:
            frames: list of dict:
                - timestamp (int)
                - image (np.ndarray)
                - exposure (int or None)  # if provided by dv
        """
        if "frames" not in self.aedat.names:
            return None

        frames = []
        for i, fr in enumerate(self.aedat["frames"]):
            # Some dv versions / devices may not expose fr.exposure
            exposure = getattr(fr, "exposure", None)

            frames.append({
                "timestamp": int(getattr(fr, "timestamp")),
                "exposure": None if exposure is None else int(exposure),
                "image": fr.image.copy()
            })

            if max_frames and i >= max_frames:
                break

        return frames

    def summary(self):
        print("-------- AEDAT4 SUMMARY --------")
        print(f"File: {self.file_path}")

        for name in self.aedat.names:
            stream = self.aedat[name]
            print(f" - Stream: {name} | type: {type(stream)}")

if __name__ == '__main__':
    aedat4file = './AreaXO/ottawa_8.aedat4'
    aedat = Aedat4Data(aedat4file, auto_load=True)
    aedat.summary()

    frames = aedat.read_frames(max_frames=2)
    print("\nSample frame info:")
    for f in frames:
        print(f["timestamp"], f["exposure"], f["image"].shape)
        
    f = frames[0]
    img = f["image"]

    print(img.shape, img.dtype, img.min(), img.max())

    # -----------------------------
    import matplotlib.pyplot as plt

    frames = aedat.read_frames(max_frames=3)
    f0 = frames[0]
    img = f0["image"]

    plt.figure()
    plt.imshow(img)  # uint8 RGB can show directly
    plt.title(f"Frame @ {f0['timestamp']} µs")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # import cv2

    # frames = aedat.read_frames(max_frames=2000)

    # for f in frames:
    #     img = f["image"]
    #     cv2.imshow("APS Video", img)
    #     key = cv2.waitKey(30)  # ~33fps
    #     if key == 27:          # ESC退出
    #         break

    # cv2.destroyAllWindows()

    # -----------------------------
    events_np = aedat.read_all_events()
    print("Total events:", len(events_np))
    print("First 5 events:")
    print(events_np[:5])

    # -----------------------------
    import numpy as np
    import matplotlib.pyplot as plt

    # H, W = 260, 346
    H, W = frames[0]["image"].shape[:2]
    # 举例：取 10ms 窗口（10000us），你也可以换成 frame 对齐的窗口
    t0 = events_np["t"].min()
    t1 = t0 + 50_000_000

    ev = events_np[(events_np["t"] >= t0) & (events_np["t"] < t1)]

    count = np.zeros((H, W), dtype=np.uint32)
    np.add.at(count, (ev["y"], ev["x"]), 1)

    # vis = np.log1p(count).astype(np.float32)
    # vis /= (vis.max() + 1e-6)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(vis, cmap="Blues")
    # plt.axis("off")
    # plt.show()

    # vis: 0~1 float32
    vis = np.log1p(count).astype(np.float32)
    mx = vis.max()
    if mx > 0:
        vis /= mx

    # Generate RGB according to FRED dataset：Background(30, 37, 52) + tail:(64, 126, 200) 
    bg = np.array([30, 37, 52], dtype=np.uint8)      # 30, 37, 52
    fg = np.array([64, 126, 200], dtype=np.uint8)    # 64, 126, 200

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:] = bg

    a = (vis * 1.0).clip(0, 1)[..., None]  # alpha 0~1
    rgb = (rgb * (1 - a) + fg * a).astype(np.uint8)

    plt.figure(figsize=(10, 5), facecolor="black")
    plt.imshow(rgb, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
