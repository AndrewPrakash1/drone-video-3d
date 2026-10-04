from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.spatial_model.training import train_manifest


parser = argparse.ArgumentParser(description="Train the OnePass metric-conditioned spatial model")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--device", default=None)
parser.add_argument("--views", type=int, default=6)
parser.add_argument("--image-size", type=int, default=128)
args = parser.parse_args()
result = train_manifest(
    args.manifest,
    args.output,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    device=args.device,
    views=args.views,
    image_size=args.image_size,
)
print(result)
