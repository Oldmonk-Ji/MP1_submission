"""CPU FP32 latency microbenchmark under evaluator conditions.

Builds random-weight student models and times predict_log_probs over the
validation windows with the same protocol as evaluate.py (batch 32, context
256, FP32, 4 threads). Used for the width-depth-latency Pareto before
committing training budget. Timing only -- no BPB is reported and no test
split is touched.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from common import load_data, windows
from student import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', required=True,
                   help='width:depth:heads specs, e.g. 208:9:8 192:10:6')
    p.add_argument('--split', default='validation')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    tokens, _ = load_data()[args.split]
    results = []
    for spec in args.models:
        width, depth, heads = (int(v) for v in spec.split(':'))
        config = dict(vocab=2048, width=width, heads=heads, depth=depth, context=256)
        model = build_model(config).eval()
        with torch.no_grad():
            for x, _ in windows(tokens, args.batch_size):
                model.predict_log_probs(x)
                break
            times = []
            for _ in range(args.runs):
                started = time.perf_counter()
                for x, _ in windows(tokens, args.batch_size):
                    model.predict_log_probs(x)
                times.append(time.perf_counter() - started)
        entry = {'spec': spec, 'width': width, 'depth': depth, 'heads': heads,
                 'params': sum(p.numel() for p in model.parameters()),
                 'times': times, 'median': statistics.median(times)}
        results.append(entry)
        print(json.dumps(entry))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
