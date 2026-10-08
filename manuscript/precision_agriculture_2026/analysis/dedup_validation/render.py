import json
from PIL import Image, ImageDraw, ImageFont
P = json.load(open('pairs.json', encoding='utf-8'))
font = ImageFont.truetype('arial.ttf', 22)
S = 300
for s0 in range(0, len(P), 10):
    sheet = Image.new('RGB', (2 * (2 * S + 40), 5 * (S + 40)), 'white'); d = ImageDraw.Draw(sheet)
    for k, p in enumerate(P[s0:s0 + 10]):
        x0 = (k % 2) * (2 * S + 40); y0 = (k // 2) * (S + 40)
        for c, path in enumerate((p['a_path'], p['b_path'])):
            try:
                im = Image.open(path).convert('RGB'); im.thumbnail((S - 6, S - 6))
            except Exception:
                im = Image.new('RGB', (S, S), 'grey')
            sheet.paste(im, (x0 + c * S + 4, y0 + 34))
        d.text((x0 + 4, y0 + 4), f"#{s0 + k}  {p['sim']:.3f}", fill='red', font=font)
        d.line([(x0 + 2 * S + 30, y0), (x0 + 2 * S + 30, y0 + S + 40)], fill='black', width=3)
    sheet.save(f'sheet_{s0 // 10:02d}.jpg', quality=88)
