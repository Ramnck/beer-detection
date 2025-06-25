# hubconf.py
import os
from pathlib import Path
import torch
import urllib.request

# 1) Declare dependencies so torch.hub knows what to install
dependencies = ["torch", "ultralytics"]
# 2) URL to your .pt file (e.g. on GitHub Releases or S3)
MODEL_URL = "https://github.com/Ramnck/pivo-segmentation/raw/refs/heads/master/weights/model.pt"

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)

def model(pretrained: bool = False, **kwargs):
    """
    Loads YOLOv9m model.
    Args:
        pretrained (bool): if True, download and load the .pt from MODEL_URL.
        **kwargs: passed to YOLO()
    Returns:
        An instance of YOLO
    """
    # 3) Determine weights path
    if pretrained:
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints" / "yolov9m-pivo"
        model_path = cache_dir / "model.pt"
        if not model_path.exists():
            print(f"Downloading weights to {model_path} ...")
            download(MODEL_URL, model_path)
    else:
        # allow user to override with weights="path/to/your.pt"
        model_path = kwargs.pop("weights", None)
        if model_path is None:
            raise ValueError("Either set pretrained=True or pass weights='<path>'")

    # 4) Import your YOLO class here
    #    adjust the path if your YOLO code lives elsewhere in your repo
    from ultralytics import YOLO

    # 5) Instantiate and return
    return YOLO(str(model_path), **kwargs)
