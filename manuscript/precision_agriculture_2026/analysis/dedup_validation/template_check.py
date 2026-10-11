"""Third check for pairs where visual rating and keypoint matching disagree: multi-scale template matching.
The image with the smaller field of view is searched inside the other one (several scales, with flips); the maximal
normalised cross-correlation (TM_CCOEFF_NORMED) is reported. Writes template_check.csv."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def gray(path, side=480):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    s = side / max(im.shape)
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def best_ncc(big, small):
    best = -1.0
    for t in (small, cv2.flip(small, 1), cv2.flip(small, 0)):
        for sc in np.linspace(0.15, 1.0, 35):
            h, w = int(t.shape[0] * sc), int(t.shape[1] * sc)
            if h < 24 or w < 24 or h > big.shape[0] or w > big.shape[1]:
                continue
            r = cv2.matchTemplate(big, cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            best = max(best, float(r.max()))
    return best


def main() -> None:
    pairs = json.loads((HERE / 'pairs.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader(open(HERE / 'keypoint_check.csv', encoding='utf-8')))
    out = []
    for r in rows:
        i = int(r['pair']); p = pairs[i]
        a, b = gray(p['a_path']), gray(p['b_path'])
        ncc = max(best_ncc(a, b), best_ncc(b, a))
        out.append([i, r['bin'], r['sift_inliers'], r['visual_same'], round(ncc, 3)])
    with open(HERE / 'template_check.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['pair', 'bin', 'sift_inliers', 'visual_same', 'template_ncc']); w.writerows(out)
    for lab in ('1', '0'):
        v = sorted(o[4] for o in out if o[3] == lab)
        print('visual', lab, 'n', len(v), 'ncc quartiles', np.percentile(v, [0, 25, 50, 75, 100]).round(3))
    for o in out:
        if (o[3] == '1' and int(o[2]) < 30) or (o[3] == '0' and int(o[2]) >= 30):
            print('disputed', o)


if __name__ == '__main__':
    main()
