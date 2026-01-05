# inference_yolov8.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# =========================
# 配置
# =========================
MODEL_PATH = "train24/weights/best.pt"  # 改成你的 best.pt 路径
SOURCE = "images"                            # 图片 / 文件夹 / 视频 / 摄像头(0)
CONF_THRES = 0.25                                 # 推荐：0.25
IOU_THRES = 0.5
IMG_SIZE = 640
SAVE_DIR = "inference_results"

# =========================
# 加载模型
# =========================
model = YOLO(MODEL_PATH)

# =========================
# 推理
# =========================
results = model.predict(
    source=SOURCE,
    conf=CONF_THRES,
    iou=IOU_THRES,
    imgsz=IMG_SIZE,
    save=True,              # 保存结果图
    save_txt=True,          # 保存 txt（YOLO 格式）
    project=SAVE_DIR,
    name="exp",
    exist_ok=True
)

print("Inference finished.")
print(f"Results saved to: {SAVE_DIR}/exp")
