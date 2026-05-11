import numpy as np
import cv2
import datetime
from zoneinfo import ZoneInfo
from dv import AedatFile

# ---------- Config ----------
AEDAT4_PATH = "G:\\filesdatastore@hotmail.com\\OneDrive\\Neuromorphic Camera\\2025-12-16 Area X.O\\Raw\\ottawa_8.aedat4"
WINDOW_US = 33_000                 # 33ms
MAX_WINDOWS = None                 # None=一直播；或填 300 之类
SHOW_POLARITY = False              # False: 只统计次数；True: 正负极性分开(可扩展)

# 你之前用的 FRED 风格配色
BG = np.array([30, 37, 52], dtype=np.uint8)
FG = np.array([64, 126, 200], dtype=np.uint8)

TZ_OTTAWA = ZoneInfo("America/Toronto")

def unix_us_to_ottawa_str(t_us: int) -> str:
    """Unix epoch microseconds -> Ottawa local time string."""
    dt_utc = datetime.datetime.fromtimestamp(t_us / 1_000_000, tz=datetime.timezone.utc)
    dt_local = dt_utc.astimezone(TZ_OTTAWA)
    # 例如：2025-12-16 13:13:33.601382 EST
    return dt_local.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt_local.microsecond:06d} " + dt_local.tzname()

def render_event_image(count: np.ndarray) -> np.ndarray:
    """
    count: HxW uint32
    return: HxWx3 uint8 (RGB-like but we'll show via cv2 as BGR)
    """
    vis = np.log1p(count).astype(np.float32)
    mx = float(vis.max())
    if mx > 0:
        vis /= mx

    a = vis[..., None]  # alpha 0~1
    rgb = (BG * (1 - a) + FG * a).astype(np.uint8)
    # cv2 uses BGR
    bgr = rgb[..., ::-1].copy()
    return bgr

def main():
    aedat = AedatFile(AEDAT4_PATH)
    if "events" not in aedat.names:
        raise RuntimeError("No 'events' stream in this aedat4.")

    # 获取分辨率：优先用 frames 的 shape；否则用你常见的 (260,346)
    H, W = 260, 346
    if "frames" in aedat.names:
        itf = iter(aedat["frames"])
        try:
            fr0 = next(itf)
            H, W = fr0.image.shape[:2]
        except StopIteration:
            pass

    count = np.zeros((H, W), dtype=np.uint32)

    it = iter(aedat["events"])
    try:
        ev0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty events stream.")

    t0 = int(ev0.timestamp)
    # 让窗口对齐到 t0（你也可以对齐到 33ms 的整数倍，但没必要）
    win_start = t0
    win_end = win_start + WINDOW_US

    # 把第一个 event 计入
    x0, y0 = int(ev0.x), int(ev0.y)
    if 0 <= x0 < W and 0 <= y0 < H:
        count[y0, x0] += 1

    win_idx = 0
    cv2.namedWindow("Event 33ms", cv2.WINDOW_NORMAL)

    for ev in it:
        t = int(ev.timestamp)

        # 把超出当前窗口的 event 触发“输出/显示”，并推进窗口
        while t >= win_end:
            # 渲染当前窗口
            img = render_event_image(count)

            # 标注 Ottawa 绝对时间：用窗口中心时间更直观
            t_center = win_start + WINDOW_US // 2
            time_str = unix_us_to_ottawa_str(t_center)

            # 也可以标注窗口范围
            range_str = f"Window: {WINDOW_US/1000:.1f} ms  |  idx: {win_idx}"

            cv2.putText(img, time_str, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, range_str, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("Event 33ms", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return

            # 清空计数并推进窗口
            count.fill(0)
            win_start = win_end
            win_end = win_start + WINDOW_US
            win_idx += 1

            if MAX_WINDOWS is not None and win_idx >= MAX_WINDOWS:
                cv2.destroyAllWindows()
                return

        # 累积当前 event 到窗口
        x, y = int(ev.x), int(ev.y)
        if 0 <= x < W and 0 <= y < H:
            count[y, x] += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
