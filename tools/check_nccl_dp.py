"""NCCL acceptance: all three tuners, token weighting, empty ranks and resume.
Requires PyTorch only on the test driver. The build/training has no torch dependency.
Examples:
  python tools/check_nccl_dp.py --build build --work /tmp/dp-qa --devices 0,1,2,3
  python tools/check_nccl_dp.py --build /home/no22/rwkv_lightning_cuda/build \
    --work /tmp/dp-qa --devices 0,1,2,3 --host no22@100.64.0.4 \
    --remote-work /home/no22/dp-qa --ssh-socket /tmp/rwkv-nccl-ssh
"""
import argparse
import json
import pathlib
import re
import shlex
import subprocess
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=pathlib.Path, required=True)
p.add_argument('--work', type=pathlib.Path, required=True)
p.add_argument('--devices', default='0,1,2,3')
p.add_argument('--host')
p.add_argument('--remote-work', type=pathlib.Path)
p.add_argument('--ssh-socket')
p.add_argument('--width', type=int, default=64,
               help='fixture width; use 256 to exercise optimized NoRA projections')
p.add_argument('--rank', type=int, default=8,
               help='adapter rank; use 16 to exercise specialized NoRA kernels')
a = p.parse_args()
if a.width < 64 or a.width % 64 or a.rank < 1:
    p.error('--width must be a positive multiple of 64; --rank must be positive')
if a.work.exists():
    p.error('--work must be a fresh directory (protect existing checkpoints)')
a.work.mkdir(parents=True)
a.build = a.build.absolute()
remote = a.remote_work if a.host else a.work.absolute()
if remote is None:
    p.error('--host requires --remote-work')
ssh = ['ssh'] + (['-S', a.ssh_socket] if a.ssh_socket else [])
def execute(cmd, timeout=180, expect=0):
    cmd = [str(x) for x in cmd]
    if a.host:
        cmd = ssh + [a.host, shlex.join(cmd)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if (result.returncode == 0) != (expect == 0):
        raise RuntimeError(result.stdout)
    return result.stdout

torch.manual_seed(1234)
# Two layers exercise v_first cross-layer gradients; nonzero mixes exercise shifts.
C, F, V, R, L = a.width, a.width * 2, 256, 16, 2
w = {}
def put(n, shape, kind='random'):
    w[n] = (torch.ones(shape) if kind == 'one' else torch.zeros(shape) if kind == 'zero'
            else torch.randn(shape) * .06).bfloat16().contiguous()
put('emb.weight', (V, C));put('head.weight', (V, C))
put('ln_out.weight',(C,), 'one');put('ln_out.bias',(C,), 'zero')
for l in range(L):
    b=f'blocks.{l}.'
    for norm in (['ln0','ln1','ln2'] if l==0 else ['ln1','ln2']):
        put(b+norm+'.weight',(C,),'one');put(b+norm+'.bias',(C,),'zero')
    for n in ['x_r','x_w','x_k','x_v','x_a','x_g']:
        w[b+'att.'+n]=torch.full((C,),.5,dtype=torch.bfloat16)
    for n in ['receptance','key','value','output']:
        put(b+'att.'+n+'.weight',(C,C))
    for n in ['w','a','g']+(['v'] if l else []):
        put(b+'att.'+n+'1',(C,R));put(b+'att.'+n+'2',(R,C))
    for n in ['w0','a0']+(['v0'] if l else []): put(b+'att.'+n,(C,),'zero')
    put(b+'att.k_k',(C,),'one');put(b+'att.k_a',(C,),'one');put(b+'att.r_k',(C//64,64))
    put(b+'att.ln_x.weight',(C,),'one');put(b+'att.ln_x.bias',(C,),'zero')
    w[b+'ffn.x_k']=torch.full((C,),.5,dtype=torch.bfloat16)
    put(b+'ffn.key.weight',(F,C));put(b+'ffn.value.weight',(C,F))
torch.save(w,a.work/'tiny.pth')
# ASCII-only fixture: first 127 vocabulary entries preserve byte token IDs.
repo=pathlib.Path(__file__).resolve().parents[1]
(a.work/'vocab.txt').write_text(''.join((repo/'assets/rwkv_vocab_v20230424.txt').read_text().splitlines(True)[:127]))
# Deliberately unequal lengths/masks, skipped rows and a 2-sample final batch.
rows = [{'text': ''}, {'segments': [{'text': 'masked', 'train': False}]}]
for i, length in enumerate([47, 9, 32, 16, 43, 21, 6, 37, 11]):
    rows.append({'segments': [{'text': 'prompt'[:i % 6], 'train': False},
                              {'text': ('abcxyz' * 9)[:length], 'train': True},
                              {'text': 'suffix', 'train': False}]})
(a.work/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
if a.host:
    execute(['mkdir', remote])
    subprocess.run(['rsync', '-az', '-e', shlex.join(ssh),
                    str(a.work) + '/', a.host + ':' + str(remote) + '/'], check=True)

def run(mode, name, devices=None, steps=2, resume=None, extra=(), expect=0):
    binary = a.build / ('rwkv_' + mode + '_tune' + ('_dp' if devices else ''))
    cmd = [binary, '--model', remote/'tiny.pth', '--vocab', remote/'vocab.txt',
           '--data', remote/'train.jsonl', '--output', remote/name,
           '--ctx', '48', '--chunk', '8', '--batch-size', '7', '--micro-bsz', '2',
           '--epochs', '2', '--lr', '.003', '--lr-final', '.003',
           '--warmup-steps', '0', '--max-steps', str(steps), '--save-every', '1']
    if mode != 'state': cmd += ['--rank', str(a.rank)]
    if devices: cmd += ['--devices', devices]
    if resume: cmd += ['--resume', remote/resume]
    output = execute([*cmd, *extra], expect=expect)
    (a.work/(name+'.log')).write_text(output)
    if a.host and expect == 0:
        subprocess.run(['rsync', '-az', '-e', shlex.join(ssh),
                        a.host + ':' + str(remote/name) + '/', str(a.work/name) + '/'], check=True)
    return [float(x) for x in re.findall(r'loss=([0-9.]+)', output)]

def compare(x, y, exact=False):
    left = torch.load(x, weights_only=True)
    right = torch.load(y, weights_only=True)
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0 if exact else .003,
                                   atol=0 if exact else 3e-5, msg=key)
    print('PASS', x.parent.name, 'vs', y.parent.name, '(exact)' if exact else '')

for mode in ['state', 'miss', 'nora']:
    base = mode + '-single'
    one = mode + '-dp1'
    many = mode + '-dp'
    ref = run(mode, base)
    for name, devices in [(one, a.devices.split(',')[0]), (many, a.devices)]:
        losses = run(mode, name, devices)
        assert len(losses) == len(ref) == 2
        assert max(abs(x-y) for x,y in zip(ref, losses)) < .001, (ref, losses)
        if mode == 'state':
            compare(a.work/base/'state-final.pth', a.work/name/'state-final.pth')
        else:
            for step in (1, 2):
                compare(a.work/base/f'checkpoint-{step}/training.pth',
                        a.work/name/f'checkpoint-{step}/training.pth')
    if mode != 'state':
        resumed = mode + '-resume'
        run(mode, resumed, a.devices, resume=many+'/checkpoint-1')
        compare(a.work/many/'checkpoint-2/training.pth',
                a.work/resumed/'checkpoint-2/training.pth', exact=True)
        run(mode, mode+'-bad-resume', a.devices, resume=many+'/checkpoint-1',
            extra=['--seed', '99'], expect=1)

run('state', 'muon-single', extra=['--optimizer', 'muon'])
run('state', 'muon-dp', a.devices, extra=['--optimizer', 'muon'])
compare(a.work/'muon-single/state-final.pth', a.work/'muon-dp/state-final.pth')
run('nora', 'duplicate-device', '0,0', expect=1)
# One rank fails before NCCL initialization: supervisor must kill/reap peers.
run('nora', 'bad-device', a.devices.split(',')[0]+',9999', expect=1)
# Only rank zero attempts checkpoint creation. Its I/O failure must terminate
# peers waiting at the checkpoint barrier rather than leave them hanging.
run('nora', 'checkpoint-io-error', a.devices,
    extra=['--output', remote/'nora-dp'], expect=1)
# Test cancellation in the machine that owns the processes, including remote
# runs; terminating just an ssh client would not exercise the supervisor.
cancel_command = [str(a.build/'rwkv_nora_tune_dp'), '--model', str(remote/'tiny.pth'),
                  '--vocab', str(remote/'vocab.txt'), '--data', str(remote/'train.jsonl'),
                  '--output', str(remote/'cancel'), '--devices', a.devices,
                  '--epochs', '100000', '--ctx', '48', '--chunk', '8']
cancel_test = r'''
import json, os, pathlib, signal, subprocess, sys, time
with open(sys.argv[2], 'w') as log:
    p = subprocess.Popen(json.loads(sys.argv[1]), stdout=log, stderr=log)
    children = []
    try:
        for _ in range(100):
            path = pathlib.Path(f'/proc/{p.pid}/task/{p.pid}/children')
            children = path.read_text().split() if path.exists() else []
            if len(children) == int(sys.argv[3]): break
            if p.poll() is not None: raise RuntimeError('launcher exited before cancellation')
            time.sleep(.05)
        assert len(children) == int(sys.argv[3]), children
        time.sleep(.2)
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=15) == 128 + signal.SIGTERM
        assert all(not pathlib.Path('/proc/'+pid).exists() for pid in children), children
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
'''
execute(['python3', '-c', cancel_test, json.dumps(cancel_command),
         remote/'cancel.log', str(len(a.devices.split(',')))])
print('PASS: all NCCL acceptance checks')
