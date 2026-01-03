# Neuromorphic-Camera
## 1. Prerequisites
py -0
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
(.venv) PS D:\GITHUB.COM\Neuromorphic-Camera> python --version
Python 3.10.11

python.exe -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install dv h5py opencv-python tqdm 
pip install -U ultralytics


cd .\datasets\
python .\1_fred_split_and_unzip_fred.py
<!-- python .\datasets\converter.py -->

yolo detect train data=FRED/fred_for_yolo/fred_yolo_rgb/data.yaml model=yolov8n.pt imgsz=640 epochs=10 batch=16 fraction=0.1 device=0

## 2. Hardware

[inivation](https://inivation.com/) 
[DAVIS 346](https://inivation.com/wp-content/uploads/2019/08/DAVIS346.pdf)


## 3. Datasets


## Reference:
https://github.com/ChenYichen9527/Ev-UAV
https://arxiv.org/abs/2506.23575

N. Chen, C. Xiao, Y. Dai, S. He, M. Li, and W. An,
"Event-based Tiny Object Detection: A Benchmark Dataset and Baseline,"
arXiv preprint arXiv:2506.23575, 2025.

