"""Validate the compiled MXQ model on a Mobilint NPU (run on the NPU machine).

Usage: python npu_validate.py <model.mxq> <val_list.txt> <images_dir>

The MXQ input is int8: the script normalizes like training, multiplies by the
model's input scales and rounds into the int8 range.
"""
import os
import sys

import maccel
import numpy as np
from PIL import Image


def preprocess(path, scale_list, size=240):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = arr[np.newaxis, ...].astype(np.float32)
    scales = np.array(scale_list, dtype=np.float32).reshape(1, 1, 1, 3)
    return np.clip(np.round(arr * scales), -128, 127).astype(np.int8)


mxq, txt, imgs = sys.argv[1], sys.argv[2], sys.argv[3]
model = maccel.load(mxq)
scale_list = model.get_input_scale()[0].scale_list
print(f"Input scales: {scale_list}")

samples = []
with open(txt) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            p = os.path.join(imgs, os.path.basename(parts[0]))
            if os.path.exists(p):
                samples.append((p, int(parts[1])))
print(f"Found {len(samples)} images")

correct = 0
for i, (path, label) in enumerate(samples):
    out = model.infer_to_float([preprocess(path, scale_list)])
    if np.argmax(out[0]) == label:
        correct += 1
    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(samples)}] Acc: {correct/(i+1)*100:.2f}%")
print(f"Final NPU Accuracy: {correct/len(samples)*100:.2f}% ({correct}/{len(samples)})")
