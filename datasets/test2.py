# plot_trajectory_from_airdata.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_col(df, candidates):
    """
    从 df.columns 里按关键词模糊匹配列名（不区分大小写）。
    candidates: e.g. ["latitude", "lat"]
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand = cand.lower()
        # 先精确匹配
        if cand in cols_lower:
            return cols_lower[cand]
        # 再子串匹配
        for k_lower, orig in cols_lower.items():
            if cand in k_lower:
                return orig
    return None

def plot_trajectory(csv_path, out_dir=".", drop_zero_zero=True):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 自动挑列（按你这份文件会匹配到：latitude, longitude, altitude_above_seaLevel(meters), time(millisecond)）
    lat_col = pick_col(df, ["latitude", "lat"])
    lon_col = pick_col(df, ["longitude", "lon", "lng", "long"])
    alt_col = pick_col(df, ["altitude_above_sealevel", "altitude", "alt", "height", "amsl", "asl", "gpsalt", "baroalt"])
    time_col = pick_col(df, ["time(millisecond)", "time", "timestamp", "datetime", "utc", "gpstime"])

    if lat_col is None or lon_col is None:
        raise ValueError(f"找不到 latitude/longitude 列：lat={lat_col}, lon={lon_col}")

    # 如果 alt 或 time 找不到，也能画（只是少了 3D 或着色）
    use_cols = [lat_col, lon_col]
    if alt_col is not None: use_cols.append(alt_col)
    if time_col is not None: use_cols.append(time_col)

    traj = df[use_cols].copy()

    # 转数值 & 清理 NaN/Inf
    for c in use_cols:
        traj[c] = pd.to_numeric(traj[c], errors="coerce")
    traj = traj.replace([np.inf, -np.inf], np.nan).dropna(subset=[lat_col, lon_col])

    # 过滤无效 GPS： (0,0)
    if drop_zero_zero:
        mask_valid = ~((traj[lat_col].abs() < 1e-9) & (traj[lon_col].abs() < 1e-9))
        traj = traj.loc[mask_valid].copy()

    print("=== Columns used ===")
    print(" lat:", lat_col)
    print(" lon:", lon_col)
    print(" alt:", alt_col)
    print(" time:", time_col)
    print(" rows:", len(traj))

    # 时间归一化用于着色
    if time_col is not None and traj[time_col].notna().any():
        t = traj[time_col].to_numpy()
        t_norm = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t) + 1e-12)
    else:
        t_norm = None

    # -------- 2D plot (Lon-Lat) --------
    plt.figure(figsize=(8, 6))
    if t_norm is not None:
        sc = plt.scatter(traj[lon_col], traj[lat_col], c=t_norm, s=6)
        plt.colorbar(sc, label="Normalized time (0=start, 1=end)")
    else:
        plt.scatter(traj[lon_col], traj[lat_col], s=6)
    plt.plot(traj[lon_col], traj[lat_col], linewidth=0.8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectory (Longitude vs Latitude)")
    plt.tight_layout()

    out2d = out_dir / "trajectory_2d.png"
    plt.savefig(out2d, dpi=200)
    plt.show()

    # -------- 3D plot (Lon-Lat-Alt) --------
    if alt_col is not None and traj[alt_col].notna().any():
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[lon_col], traj[lat_col], traj[alt_col], linewidth=1.0)

        # 起点/终点标记
        ax.scatter(traj[lon_col].iloc[0], traj[lat_col].iloc[0], traj[alt_col].iloc[0], s=40, marker="o")
        ax.scatter(traj[lon_col].iloc[-1], traj[lat_col].iloc[-1], traj[alt_col].iloc[-1], s=40, marker="^")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (m)")
        ax.set_title("3D Trajectory (Lon, Lat, Alt)")
        plt.tight_layout()

        out3d = out_dir / "trajectory_3d.png"
        plt.savefig(out3d, dpi=200)
        plt.show()
    else:
        print("⚠️ 没找到可用的 altitude 列，跳过 3D 图。")

    print(f"Saved:\n - {out2d}")
    if alt_col is not None:
        print(f" - {out_dir / 'trajectory_3d.png'} (if altitude exists)")

if __name__ == "__main__":
    # 改成你的文件路径即可
    plot_trajectory(
        csv_path="G:\\filesdatastore@hotmail.com\\OneDrive\\Neuromorphic Camera\\2025-12-16 Area X.O\\Processed data\\FlightRecord\\Dec-16th-2025-01-10PM-Flight-Airdata.csv",
        out_dir=".",
        drop_zero_zero=True
    )
