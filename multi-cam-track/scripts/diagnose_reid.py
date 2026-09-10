"""Inspect saved ReID embeddings without running detection or inference."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/epfl_4p_v2')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    records = []
    for path in sorted((Path(args.output_dir) / 'features').glob('*.npz')):
        data = np.load(path, allow_pickle=False)
        if 'reid' not in data: continue
        v = data['reid'].astype(np.float32); v /= np.linalg.norm(v) + 1e-8
        records.append((str(data['camera_id']), int(data['local_track_id']), int(data['global_id']), v, path.name))
    scores, pairs = [], []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a[0] == b[0]: continue
            score = float(np.dot(a[3], b[3]))
            scores.append(score); pairs.append((score, a, b))
    if not scores:
        raise SystemExit('No cross-camera ReID embeddings found.')
    values = np.asarray(scores)
    print(f'Embeddings: {len(records)}; cross-camera pairs: {len(values)}')
    print('Similarity percentiles:', ', '.join(f'p{q}={np.percentile(values, q):.3f}' for q in (50, 75, 90, 95, 99)))
    print(f'Max similarity: {values.max():.3f}')
    print('\nHighest cross-camera similarities:')
    for score, a, b in sorted(pairs, reverse=True, key=lambda x: x[0])[:args.top]:
        print(f'  {score:.3f}  {a[0]}:{a[1]} (G{a[2]})  <->  {b[0]}:{b[1]} (G{b[2]})')


if __name__ == '__main__':
    main()
