"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def kd_loss(student_logits, targets, teacher, ids, alpha, temperature, teacher_temperature, device, precision,
            temperature_mode='legacy_shared'):
    """L = (1-alpha)*CE + alpha*T^2*KL with a frozen teacher evaluated online."""
    with torch.no_grad():
        with autocast(device, precision):
            teacher_logits = teacher.logits(ids) if hasattr(teacher, 'logits') else teacher(ids).float()
    t_eff = teacher_temperature * temperature
    ce = F.cross_entropy(student_logits.flatten(0, 1), targets)
    if temperature_mode == 'teacher_calibrated':
        # Match the calibrated teacher distribution: only the teacher receives
        # its inference calibration. T is shared for distillation, with T^2 scaling.
        teacher_logp = F.log_softmax(teacher_logits.float() / t_eff, dim=-1).flatten(0, 1)
        student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1).flatten(0, 1)
        kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).mean()
        return (1 - alpha) * ce + alpha * temperature * temperature * kl
    if temperature_mode != 'legacy_shared':
        raise ValueError('Unknown KD temperature mode')
    teacher_probs = F.softmax(teacher_logits / t_eff, dim=-1).flatten(0, 1)
    student_logp = F.log_softmax(student_logits / t_eff, dim=-1).flatten(0, 1)
    kl = (teacher_probs * (teacher_probs.log() - student_logp)).sum(dim=-1).mean()
    return (1 - alpha) * ce + alpha * t_eff * t_eff * kl


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw',
                   help='muon: hidden >=2D matrices use Muon (tools/muon.py), the rest auxiliary Adam.')
    p.add_argument('--aux-lr-scale', type=float, default=1.,
                   help='For --optimizer muon: scale of the auxiliary Adam groups relative to --lr.')
    p.add_argument('--hidden-lr', type=float,
                   help='Muon peak lr for hidden matrices; defaults to --lr.')
    p.add_argument('--aux-lr', type=float,
                   help='Muon auxiliary Adam peak lr; overrides --lr * --aux-lr-scale.')
    p.add_argument('--muon-momentum', type=float, default=.95)
    p.add_argument('--muon-scale-mode', choices=['legacy', 'match_rms'], default='legacy',
                   help='legacy reproduces old runs; match_rms gives all matrix shapes equal update RMS.')
    p.add_argument('--muon-rms-scale', type=float, default=.2,
                   help='Target pre-lr update RMS used by --muon-scale-mode match_rms.')
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--schedule', choices=['cosine_oscillation', 'sgdr', 'wsd'],
                   default='cosine_oscillation',
                   help='cosine_oscillation exactly preserves the historical schedule.')
    p.add_argument('--cycles', type=float, default=1.,
                   help='Oscillation multiplier, or number of complete cycles for SGDR.')
    p.add_argument('--wsd-decay-steps', type=int, default=2400,
                   help='For WSD: number of final cosine-decay steps after the stable phase.')
    p.add_argument('--label-smoothing', type=float, default=0.)
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Frozen KD teacher, evaluated online on each training batch.')
    p.add_argument('--kd-alpha', type=float, default=0.,
                   help='KD loss weight in (0, 1]; 0 disables distillation.')
    p.add_argument('--kd-temperature', type=float, default=1.,
                   help='KD temperature, applied on top of the teacher ensemble temperature.')
    p.add_argument('--kd-temperature-mode', choices=['legacy_shared', 'teacher_calibrated'],
                   default='legacy_shared', help='Whether teacher inference calibration also scales student logits.')
    p.add_argument('--init-checkpoint', type=Path,
                   help='Optional checkpoint whose weights initialize the student (fresh optimizer).')
    p.add_argument('--dropout', type=float,
                   help='Optional dropout override; defaults to the value in --config.')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='EMA weight averaging decay (0 disables); ema.pt is saved at the end.')
    p.add_argument('--ema-decays', type=float, nargs='+',
                   help='Maintain multiple EMA tracks in one run; saves ema_<decay>.pt files.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.cycles <= 0:
        p.error('--cycles must be positive.')
    if args.schedule == 'wsd' and not (1 <= args.wsd_decay_steps <= args.steps-args.warmup_steps):
        p.error('WSD requires 1 <= --wsd-decay-steps <= steps - warmup_steps.')
    if args.ema_decay < 0 or args.ema_decay >= 1:
        p.error('--ema-decay must be in [0, 1).')
    if args.ema_decays and any(d <= 0 or d >= 1 for d in args.ema_decays):
        p.error('--ema-decays values must be in (0, 1).')
    if args.dropout is not None and not (0 <= args.dropout < 1):
        p.error('--dropout must be in [0, 1).')
    if args.teacher_checkpoint is not None and not (0 < args.kd_alpha <= 1):
        p.error('--teacher-checkpoint requires --kd-alpha in (0, 1].')
    if args.kd_alpha != 0 and args.teacher_checkpoint is None:
        p.error('--kd-alpha requires --teacher-checkpoint.')
    if args.kd_alpha != 0 and args.kd_temperature <= 0:
        p.error('--kd-temperature must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    if args.dropout is not None:
        config['dropout'] = args.dropout
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.init_checkpoint is not None:
        init_ck = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if init_ck['protocol'] != PROTOCOL:
            raise ValueError('Init checkpoint belongs to a different course protocol.')
        if init_ck['config'].get('vocab') != config.get('vocab') or \
           init_ck['config'].get('context') != config.get('context'):
            raise ValueError('Init checkpoint must share vocab and context with the student.')
        model.load_state_dict(init_ck['model'])
        print(json.dumps({'init': str(args.init_checkpoint)}), flush=True)
    teacher = None
    teacher_temperature = 1.
    if args.teacher_checkpoint is not None:
        teacher_ck = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=True)
        if teacher_ck['protocol'] != PROTOCOL:
            raise ValueError('Teacher checkpoint belongs to a different course protocol.')
        if teacher_ck['config'].get('vocab') != config.get('vocab') or \
           teacher_ck['config'].get('context') != config.get('context'):
            raise ValueError('Teacher and student must share vocab and context.')
        teacher, _ = make_model(teacher_ck['implementation'], teacher_ck['config'], device)
        teacher.load_state_dict(teacher_ck['model'])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher_temperature = float(teacher_ck['config'].get('ensemble', {}).get('temperature', 1.0))
        print(json.dumps({'teacher': {'checkpoint': str(args.teacher_checkpoint),
                                      'parameters': sum(p.numel() for p in teacher.parameters()),
                                      'temperature': teacher_temperature}}), flush=True)
    hidden_lr = args.hidden_lr if args.hidden_lr is not None else args.lr
    aux_lr = args.aux_lr if args.aux_lr is not None else args.lr * args.aux_lr_scale
    if args.optimizer == 'muon':
        from tools.muon import SingleDeviceMuonWithAuxAdam
        hidden = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2]
        aux = [p for n, p in model.named_parameters() if 'blocks' not in n or p.ndim < 2]
        optimizer = SingleDeviceMuonWithAuxAdam([
            dict(params=hidden, lr=hidden_lr, momentum=args.muon_momentum,
                 weight_decay=args.weight_decay, scale_mode=args.muon_scale_mode,
                 rms_scale=args.muon_rms_scale, use_muon=True),
            dict(params=aux, lr=aux_lr, betas=(.9, .95), eps=1e-10,
                 weight_decay=args.weight_decay, use_muon=False)])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_bpb = None
    recipe = {'optimizer':args.optimizer,'lr':args.lr,'hidden_lr':hidden_lr if args.optimizer == 'muon' else None,
              'aux_lr':aux_lr if args.optimizer == 'muon' else None,
              'aux_lr_scale':args.aux_lr_scale if args.optimizer == 'muon' else None,
              'muon_momentum':args.muon_momentum if args.optimizer == 'muon' else None,
              'muon_scale_mode':args.muon_scale_mode if args.optimizer == 'muon' else None,
              'muon_rms_scale':args.muon_rms_scale if args.optimizer == 'muon' else None,
              'weight_decay':args.weight_decay,'warmup_steps':args.warmup_steps,
              'min_lr_ratio':args.min_lr_ratio,'schedule':args.schedule,'cycles':args.cycles,
              'wsd_decay_steps':args.wsd_decay_steps if args.schedule == 'wsd' else None,
              'label_smoothing':args.label_smoothing,'dropout':config.get('dropout'),
              'teacher_checkpoint':str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
              'teacher_temperature':teacher_temperature,
              'kd_alpha':args.kd_alpha,'kd_temperature':args.kd_temperature,
              'kd_temperature_mode':args.kd_temperature_mode,
              'init_checkpoint':str(args.init_checkpoint) if args.init_checkpoint else None,
              'ema_decay':args.ema_decay,
              'ema_decays':args.ema_decays,
              'batch_size':args.batch_size,'steps':args.steps}
    ema_decays = list(dict.fromkeys(args.ema_decays or ([args.ema_decay] if args.ema_decay > 0 else [])))
    ema_states = {decay:{k:v.detach().float().clone() for k,v in model.state_dict().items()}
                  for decay in ema_decays}
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        warmup_scale = min(1., (step+1)/args.warmup_steps)
        if args.schedule == 'cosine_oscillation':
            lr_shape = (args.min_lr_ratio + (1-args.min_lr_ratio)*.5
                        * (1+math.cos(math.pi*step*args.cycles/args.steps)))
        elif args.schedule == 'sgdr':
            cycle_steps = args.steps / args.cycles
            phase = (step % cycle_steps) / cycle_steps
            lr_shape = (args.min_lr_ratio + (1-args.min_lr_ratio)*.5
                        * (1+math.cos(math.pi*phase)))
        else:
            decay_steps = min(args.wsd_decay_steps, args.steps-args.warmup_steps)
            stable_end = args.steps - decay_steps
            if step < stable_end:
                lr_shape = 1.
            else:
                decay_progress = (step-stable_end) / max(1, decay_steps-1)
                lr_shape = (args.min_lr_ratio + (1-args.min_lr_ratio)*.5
                            * (1+math.cos(math.pi*decay_progress)))
        schedule_scale = warmup_scale * lr_shape
        for group in optimizer.param_groups:
            if args.optimizer == 'muon':
                peak_lr = aux_lr if group.get('use_muon') is False else hidden_lr
            else:
                peak_lr = args.lr
            group['lr'] = peak_lr * schedule_scale
        optimizer.zero_grad(set_to_none=True)
        ids, targets = batch[:, :-1], batch[:, 1:].flatten()
        with autocast(device, precision):
            student_logits = model(ids).float()
            if teacher is None:
                loss = F.cross_entropy(student_logits.flatten(0, 1), targets,
                                       label_smoothing=args.label_smoothing)
            else:
                loss = kd_loss(student_logits, targets, teacher, ids, args.kd_alpha,
                               args.kd_temperature, teacher_temperature, device, precision,
                               args.kd_temperature_mode)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if ema_states:
            with torch.no_grad():
                for decay, ema_state in ema_states.items():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(decay).add_(v.float(), alpha=1-decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if best_bpb is None or intermediate['bpb'] < best_bpb:
                best_bpb = intermediate['bpb']
                torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                            'model':model.cpu().state_dict(),'seed':args.seed,
                            'train_tokens':(step+1)*args.batch_size*256,'recipe':recipe},
                           args.run_dir/'best.pt')
                print(json.dumps({'best':{'step':step+1,'bpb':best_bpb}}),flush=True)
                model.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,'recipe':recipe},checkpoint)
    for decay, ema_state in ema_states.items():
        ema_name = 'ema.pt' if len(ema_states) == 1 and not args.ema_decays else f'ema_{decay:g}.pt'
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                    'model':{k: v.cpu() for k, v in ema_state.items()},'seed':args.seed,
                    'train_tokens':args.steps*args.batch_size*256,
                    'ema_decay':decay,'recipe':recipe}, args.run_dir/ema_name)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'recipe':recipe,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
