"""
Lightweight utility to extract coarse data from chart-like images.
Currently computes a normalized color histogram as a proxy for data distribution.
"""
from typing import Dict, Any, List

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


def extract_chart_data(image_path: str, bins: int = 16) -> Dict[str, Any]:
    """
    Extract a coarse histogram from an image (RGB), useful as a proxy for chart data.
    Falls back to empty result if Pillow is unavailable.
    """
    if Image is None:
        return {"bins": bins, "histogram": [], "note": "Pillow not installed"}
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    hist = img.histogram()
    # group histogram into bins per channel
    per_channel: List[List[float]] = []
    for c in range(3):
        channel = hist[c * 256 : (c + 1) * 256]
        bin_size = 256 // bins
        binned = [sum(channel[i : i + bin_size]) for i in range(0, 256, bin_size)]
        total = sum(binned) or 1
        per_channel.append([round(x / total, 6) for x in binned])
    return {"bins": bins, "histogram": per_channel, "width": img.width, "height": img.height}
