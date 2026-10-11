"""Image preparation for the multimodal expert (public version).

`prepare(path)` re-encodes a photograph exactly as it was sent to the multimodal model: EXIF orientation applied, converted
to a fresh RGB JPEG (quality 92, no metadata), downscaled to at most ~1.05 megapixels, plus four quadrant tiles for
photographs larger than 4.2 megapixels. Each encoded image is stored under outputs/mainline_v2/images/<sha256>.jpg and the
list of hashes is returned; these hashes are the keys of the cached responses.
"""
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / 'outputs/mainline_v2/images'
BUDGET_PX, TILE_TRIGGER_PX = 1_050_000, 4_200_000


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _encode(im: Image.Image) -> str:
    w, h = im.size
    if w * h > BUDGET_PX:
        s = (BUDGET_PX / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=92)  # fresh RGB image: no EXIF/ICC metadata carried
    b = buf.getvalue()
    h = sha_bytes(b)
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f'{h}.jpg'
    if not p.exists():
        p.write_bytes(b)
    return h


def prepare(path: str) -> list:
    im = ImageOps.exif_transpose(Image.open(path))
    im = Image.frombytes('RGB', im.size, im.convert('RGB').tobytes())
    out = [_encode(im)]
    w, h = im.size
    if w * h > TILE_TRIGGER_PX:
        for box in [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]:
            out.append(_encode(im.crop(box)))
    return out
