"""Pack two checkpoints into one ensemble checkpoint (log-prob averaging).

Member weights get 'm0.' / 'm1.' prefixes; member configs are stored in
config['ensemble']['members']. The output is a normal checkpoint that the
fixed evaluator can load with --implementation student.

Usage: python tools/make_ensemble.py a.pt b.pt -o ensemble.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+', type=Path)
    p.add_argument('-o', '--output', required=True, type=Path)
    p.add_argument('--weights', type=float, nargs='+',
                   help='Per-member logit weights (default: equal).')
    p.add_argument('--temperature', type=float,
                   help='Optional positive temperature applied after logit averaging.')
    args = p.parse_args()
    checkpoints = [torch.load(c, map_location='cpu', weights_only=True) for c in args.checkpoints]
    base = checkpoints[0]
    config = dict(base['config'])
    config['ensemble'] = {'members': [c['config'] for c in checkpoints]}
    if args.weights:
        assert len(args.weights) == len(checkpoints), 'one weight per checkpoint'
        config['ensemble']['weights'] = list(args.weights)
    if args.temperature is not None:
        if args.temperature <= 0:
            p.error('--temperature must be positive')
        config['ensemble']['temperature'] = args.temperature
    state = {}
    for i, c in enumerate(checkpoints):
        state.update({f'm{i}.{k}': v for k, v in c['model'].items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': base['protocol'], 'implementation': base['implementation'],
                'config': config, 'model': state, 'seed': base['seed'],
                'train_tokens': base['train_tokens'],
                'ensemble_from': [str(c) for c in args.checkpoints]}, args.output)
    print(f'Ensemble of {len(checkpoints)} checkpoints -> {args.output}')


if __name__ == '__main__':
    main()
