from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.spatial_model.synthetic import generate_dataset


parser = argparse.ArgumentParser(description="Generate a small synthetic OnePass spatial-model dataset")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scenes", type=int, default=12)
parser.add_argument("--views", type=int, default=6)
parser.add_argument("--size", type=int, default=128)
args = parser.parse_args()
manifest = generate_dataset(args.output, scenes=args.scenes, views=args.views, size=args.size)
print(f"wrote {len(manifest.samples)} samples to {args.output / 'manifest.json'}")
