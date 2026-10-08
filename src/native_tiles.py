"""Fixed nonoverlapping native-image quadrants."""
from PIL import Image


def quadrants(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    if min(width, height) < 2:
        raise ValueError('Quadrants require at least two pixels per axis')
    x, y = width // 2, height // 2
    return [image.crop(box) for box in ((0, 0, x, y), (x, 0, width, y),
                                       (0, y, x, height), (x, y, width, height))]
