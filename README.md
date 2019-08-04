# Pytorch InsightFace

Pretrained ResNet models from [deepinsight/insightface](https://github.com/deepinsight/insightface)
ported to pytorch.

| Model      | LFW(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)   |
| ---------- | ------ | --------- | ----------- | ------------- |
| iresnet34  | 99.65  | 92.12     | 97.70       | 96.70         |
| iresnet50  | 99.80  | 92.74     | 97.76       | 97.64         |
| iresnet100 | 99.77  | 98.27     | 98.28       | 98.47         |

## Installation

```bash
pip install git+https://github.com/nizhib/pytorch-insightface
```

## Usage

```python
import torch
from imageio import imread
from torchvision import transforms

import insightface

embedder = insightface.iresnet100(pretrained=True)
embedder.eval()

mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

face = imread('resource/sample.jpg')

tensor = preprocess(face)

with torch.no_grad():
    features = embedder(tensor.unsqueeze(0))[0]

print(features[:5])
```

## Recreating the weights locally

Download the original [insightface zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) weights and place `*.params` and `*.json` files to `resource/{model}`.

Run `python scripts/convert.py` to convert and test pytorch weights.
