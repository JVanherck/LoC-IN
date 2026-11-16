#!/usr/bin/env python3
"""Export an EigenPlaces model to TorchScript for use with libtorch.

Downloads the pretrained EigenPlaces model (ResNet50, 512-dim) from
PyTorch Hub and traces it to a TorchScript file that can be loaded in
C++ via torch::jit::load().

Usage:
    python export_eigenplaces.py [output_path]

The default output is eigenplaces_resnet50_512.pt in the current directory.

Preprocessing required at inference time (C++ side):
    - Resize to 480x480 with bicubic interpolation
    - Convert BGR (OpenCV) -> RGB
    - Scale pixel values to [0, 1]
    - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Shape: (1, 3, 480, 480)
"""

import sys
import torch


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "eigenplaces_resnet50_512.pt"

    print("Loading EigenPlaces model from PyTorch Hub...")
    model = torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=512,
    )
    model.eval()

    # Trace with a dummy input (batch=1, 3 channels, 480x480)
    dummy_input = torch.randn(1, 3, 480, 480)
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    # Verify output
    with torch.no_grad():
        out = traced(dummy_input)
    print(f"Output shape: {out.shape}")  # Expected: (1, 512)
    print(f"Output norm:  {out.norm().item():.4f}")  # Expected: ~1.0 (L2-normalised)

    traced.save(output_path)
    print(f"Saved TorchScript model to: {output_path}")


if __name__ == "__main__":
    main()
