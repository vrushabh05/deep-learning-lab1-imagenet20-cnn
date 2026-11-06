"""Export the trained model to ONNX (opset 11) and check it with ONNX Runtime."""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import ImageNet20Net

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="outputs/best_model.pth")
ap.add_argument("--out", default="outputs/model_imagenet20.onnx")
ap.add_argument("--size", type=int, default=240)
args = ap.parse_args()

model = ImageNet20Net(20)
model.load_state_dict(torch.load(args.weights, map_location="cpu"))
model.eval()
dummy = torch.randn(1, 3, args.size, args.size)
torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["logits"],
                  opset_version=11, do_constant_folding=True, dynamic_axes=None, dynamo=False)
onnx.checker.check_model(onnx.load(args.out))
sess = ort.InferenceSession(args.out)
out = sess.run(None, {"input": dummy.numpy()})[0]
with torch.no_grad():
    ref = model(dummy).numpy()
print("ONNX valid. Max abs diff vs PyTorch:", float(np.abs(out - ref).max()))
