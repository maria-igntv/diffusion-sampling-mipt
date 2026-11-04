"""PNG conversion is shared by FID samples and visual previews."""

import math
from pathlib import Path

from PIL import Image
import torch


def to_uint8(samples):
    # Matches Diffusers' [-1,1] -> [0,1] -> uint8 rounded image convention.
    return ((samples / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def export_images(shards, output, preview_count=64):
    output = Path(output)
    directory = output / "images"
    directory.mkdir(exist_ok=True)
    previews = []
    for shard in shards:
        images = to_uint8(shard["samples"])
        for offset, array in enumerate(images):
            image = Image.fromarray(array)
            path = directory / f"{shard['start'] + offset:08d}.png"
            temporary = path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(path)
            if len(previews) < preview_count:
                previews.append(image)
    if previews:
        columns = min(8, len(previews))
        grid = Image.new("RGB", (columns * 32, math.ceil(len(previews) / columns) * 32))
        for index, image in enumerate(previews):
            grid.paste(image, ((index % columns) * 32, (index // columns) * 32))
        grid.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST).save(output / "preview.png")
