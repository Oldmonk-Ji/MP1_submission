"""Validation-only temperature sweep; reuse each batch's logits, export one candidate.

Supports plain student GPT checkpoints without cache or ensemble. The exported
one-member ensemble is scored again with the unchanged official evaluator.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from common import PROTOCOL, load_data, make_model, setup, sha, windows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1., 1.04, 1.08, 1.12])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if any(not math.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error('temperatures must be finite and positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory')
    device, _ = setup(args.device, 'fp32', 4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ck['protocol'] != PROTOCOL or ck['implementation'] != 'student':
        parser.error('Expected a student checkpoint with the current protocol')
    if ck['config'].get('ensemble') or ck['config'].get('cache'):
        parser.error('Expected a single model without a cache')
    model, implementation_sha = make_model('student', ck['config'], device)
    model.load_state_dict(ck['model'])
    model.eval()
    tokens, byte_count = load_data()['validation']
    totals = {t: 0. for t in args.temperatures}
    for x, y in windows(tokens, 32):
        x, y = x.to(device), y.to(device)
        logits = model(x).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite logits')
        for t in totals:
            logp = torch.log_softmax(logits / t, dim=-1)
            losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            totals[t] += losses.masked_fill(y == -100, 0).double().sum().item()
    rows = [{'temperature': t, 'bpb': nll / math.log(2) / byte_count} for t, nll in totals.items()]
    best = min(rows, key=lambda row: row['bpb'])
    config = copy.deepcopy(ck['config'])
    config['ensemble'] = {'members': [copy.deepcopy(ck['config'])], 'temperature': best['temperature']}
    output = dict(ck)
    output.update(config=config, model={f'm0.{k}': v for k, v in ck['model'].items()},
                  ensemble_from=[str(args.checkpoint)], source_sha256=sha(args.checkpoint))
    args.output_dir.mkdir(parents=True)
    torch.save(output, args.output_dir / 'best.pt')
    result = {'split': 'validation', 'precision': 'fp32', 'device': str(device),
              'source': str(args.checkpoint), 'source_sha256': sha(args.checkpoint),
              'implementation_sha256': implementation_sha, 'rows': rows, 'best': best,
              'checkpoint_sha256': sha(args.output_dir / 'best.pt')}
    (args.output_dir / 'grid.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
