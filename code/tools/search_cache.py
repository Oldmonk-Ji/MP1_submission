"""Grid-search the within-window neural cache hyperparameters on validation.

The model weights are frozen; only the inference-time cache mixing weight
alpha and temperature theta are selected on the validation split.

Usage: python tools/search_cache.py --checkpoint runs/v3-98m/best.pt --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from common import load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--alphas', default='0.05,0.1,0.2,0.3,0.4')
    p.add_argument('--thetas', default='2,4,8')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    device, precision = setup(args.device, 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = load_data()
    rows = []
    for alpha in [float(x) for x in args.alphas.split(',')]:
        for theta in [float(x) for x in args.thetas.split(',')]:
            config = dict(checkpoint['config'])
            config['cache'] = {'alpha': alpha, 'theta': theta}
            model, _ = make_model(checkpoint['implementation'], config, device)
            model.load_state_dict(checkpoint['model'])
            result = score(model, *data['validation'], device, precision)
            rows.append({'alpha': alpha, 'theta': theta, 'bpb': result['bpb'],
                         'seconds': result['seconds']})
            print(f"alpha={alpha:<5} theta={theta:<3} bpb={result['bpb']:.4f} "
                  f"({result['seconds']:.1f}s)", flush=True)
    rows.sort(key=lambda r: r['bpb'])
    print('\nBest:', json.dumps(rows[0]))
    out = args.output or args.checkpoint.parent / 'cache_search.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
