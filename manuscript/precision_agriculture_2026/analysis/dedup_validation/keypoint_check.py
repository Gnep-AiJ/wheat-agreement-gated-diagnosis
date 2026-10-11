"""Independent, automatic second check of the visual near-duplicate ratings (ratings.csv).

For each rated pair, SIFT keypoints are matched (Lowe ratio 0.75) and a RANSAC homography is fitted; the second image is
also tried horizontally and vertically flipped. A pair is called the same photograph when at least 30 matches are
geometric inliers. Writes keypoint_check.csv and prints the agreement with the visual ratings per similarity band.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MIN_INLIERS = 30


def load(path, side=800):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    s = side / max(im.shape)
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def inliers(a, b, sift, bf):
    ka, da = sift.detectAndCompute(a, None); kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (p for p in bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good) if len(good) >= MIN_INLIERS else 0
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0


def main() -> None:
    pairs = json.loads((HERE / 'pairs.json').read_text(encoding='utf-8'))
    rated = {int(r['pair']): int(r['same_photograph']) for r in csv.DictReader(open(HERE / 'ratings.csv', encoding='utf-8'))}
    sift = cv2.SIFT_create(nfeatures=3000); bf = cv2.BFMatcher(cv2.NORM_L2)
    out = []; agree = defaultdict(lambda: [0, 0]); conf = defaultdict(int)
    for i, p in enumerate(pairs):
        a, b = load(p['a_path']), load(p['b_path'])
        n = 0 if a is None or b is None else max(inliers(a, x, sift, bf) for x in (b, cv2.flip(b, 1), cv2.flip(b, 0)))
        auto = int(n >= MIN_INLIERS)
        out.append([i, p['bin'], p['sim'], n, auto, rated[i]])
        agree[p['bin']][0] += auto == rated[i]; agree[p['bin']][1] += 1; conf[(rated[i], auto)] += 1
    with open(HERE / 'keypoint_check.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['pair', 'bin', 'similarity', 'sift_inliers', 'auto_same', 'visual_same']); w.writerows(out)
    summ = {b: dict(agree=v[0], n=v[1]) for b, v in sorted(agree.items())}
    summ['confusion_visual_auto'] = {f'visual{a}_auto{b}': c for (a, b), c in sorted(conf.items())}
    summ['total_agreement'] = sum(r[4] == r[5] for r in out) / len(out)
    (HERE / 'keypoint_check_summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    for r in out:
        if r[4] != r[5]:
            print('disagree', r)


if __name__ == '__main__':
    main()
