#!/usr/bin/env python3

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5n - yolov5x6, custom

# Images
#img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
img = './2shot.png'

# Inference
results = model(img, augment=True)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()
