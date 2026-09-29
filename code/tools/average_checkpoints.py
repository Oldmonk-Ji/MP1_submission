"""Average the model weights of several checkpoints (self-trained weight averaging).

All checkpoints must share the same config. Output keeps the first
checkpoint's protocol/implementation/config fields.

Usage: python tools/average_checkpoints.py a.pt b.pt -o averaged.pt
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
    args = p.parse_args()
    checkpoints = [torch.load(c, map_location='cpu', weights_only=True) for c in args.checkpoints]
    base = checkpoints[0]
    assert all(c['config'] == base['config'] for c in checkpoints), 'configs differ'
    names = set(base['model'])
    assert all(set(c['model']) == names for c in checkpoints), 'weight keys differ'
    averaged = {k: sum(c['model'][k].float() for c in checkpoints) / len(checkpoints)
                for k in names}
    torch.save({'protocol': base['protocol'], 'implementation': base['implementation'],
                'config': base['config'], 'model': averaged, 'seed': base['seed'],
                'train_tokens': base['train_tokens'],
                'averaged_from': [str(c) for c in args.checkpoints]}, args.output)
    print(f'Averaged {len(checkpoints)} checkpoints -> {args.output}')


if __name__ == '__main__':
    main()
