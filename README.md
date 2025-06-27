# pivo-detection

Detection model for detecting pivo (beer) objects in images.
Easily load and use the pretrained model with PyTorch Hub.

---

## Installation

Make sure you have Python 3.8+ and [PyTorch](https://pytorch.org/) installed.

```bash
pip install torch torchvision ultralytics
```

---

## Usage

You can load the model directly with PyTorch Hub:

```python
import torch

model = torch.hub.load(
    'Ramnck/pivo-detection',
    'model',
    pretrained=True,  # Download weights automatically
)

# Example: Inference on an image
# img = ... (H, W, 3) numpy or PIL.Image or torch tensor
results = model(img)
# results contains predicted masks/bboxes/scores
```

---

## Input Format

* **Image**: Accepts PIL.Image, numpy array (H, W, 3), or torch tensor (C, H, W)
* **Output**: Dict or object containing bounding boxes and confidence scores.

---

## Model Details

* Model architecture: \[YOLO/UNet/other — fill here]
* Training dataset: \[describe dataset briefly]
* Primary task: detection for "pivo" (beer) objects

---

## Example

```python
from PIL import Image
img = Image.open('your_image.jpg')
result = model(img)
print(result)
```

---

## Repository Structure

* `weights/`: Pretrained model weights (downloaded automatically)
* `hubconf.py`: PyTorch Hub entry point
* `README.md`: This file
* .ipynb's are needed for training
* dataset contains in datasets/yolo.zip in ultralytics suitable format

---

## Citation

If you use this work, please cite:

```
@misc{pivo-detection,
  author = {Ramnck},
  title = {pivo-detection},
  year = {2025},
  howpublished = {\url{https://github.com/Ramnck/pivo-detection}}
}
```

---

## License

[MIT License](LICENSE)

---

## Contact

Questions? Open an issue or contact [Ramnck](https://github.com/Ramnck).
