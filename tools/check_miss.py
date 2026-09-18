"""Deterministic MiSS GPU acceptance; requires PyTorch, builds remain torch-free.
Usage: python tools/check_miss.py --build /tmp/rwkv-miss-build --work /tmp/miss-qa
"""
import argparse
import json
import pathlib
import subprocess
import torch

p = argparse.ArgumentParser()
p.add_argument('--build', type=pathlib.Path, required=True)
p.add_argument('--work', type=pathlib.Path, required=True)
a = p.parse_args()
a.work.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1234)
# Independent explicit fixed A autograd reference, including ragged blocks.
X = (torch.randn(17, 67) * .1).half().float().requires_grad_()
D = (torch.randn(35, 16) * .1).half().float().requires_grad_()
W = (torch.randn(35, 67) * .1).half().float()
G = (torch.randn(17, 35) * .1).half().float()
A = torch.eye(16).repeat(1, 5)[:, :67]
base = X @ W.T
Y = base + X @ A.T @ D.T
(Y * G).sum().backward()
torch.save(dict(X=X.detach(), D=D.detach(), G=G, base_Y=base.detach().half(),
                base_dX=(G @ W).half(), Y=(base.detach().half().float() + X.detach() @ A.T @ D.detach().T),
                dD=D.grad, dX=(G @ W).half().float() + G @ D.detach() @ A), a.work/'reference.pth')
subprocess.run([a.build/'test/rwkv_miss_test', a.work/'reference.pth'], check=True)
# Two layers exercise v_first cross-layer gradients; nonzero mixes exercise shifts.
C, F, V, R, L = 64, 128, 256, 16, 2
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
    put(b+'att.k_k',(C,),'one');put(b+'att.k_a',(C,),'one');put(b+'att.r_k',(1,64))
    put(b+'att.ln_x.weight',(C,),'one');put(b+'att.ln_x.bias',(C,),'zero')
    w[b+'ffn.x_k']=torch.full((C,),.5,dtype=torch.bfloat16)
    put(b+'ffn.key.weight',(F,C));put(b+'ffn.value.weight',(C,F))
torch.save(w,a.work/'tiny.pth')
# ASCII-only fixture: first 127 vocabulary entries preserve byte token IDs.
repo=pathlib.Path(__file__).resolve().parents[1]
(a.work/'vocab.txt').write_text(''.join((repo/'assets/rwkv_vocab_v20230424.txt').read_text().splitlines(True)[:127]))
(a.work/'train.jsonl').write_text((json.dumps({'text':'abcabcabcabcabcabcabcabcabcabcabc'})+'\n')*4)
def run(out, chunk, steps, resume=None, batch=1, extra=()):
    cmd=[str(a.build/'rwkv_miss_tune'),'--model',str(a.work/'tiny.pth'),'--data',str(a.work/'train.jsonl'),
         '--vocab',str(a.work/'vocab.txt'),'--output',str(a.work/out),'--rank','8','--ctx','30','--chunk',str(chunk),
         '--epochs','4','--lr','.003','--lr-final','.003','--warmup-steps','0','--max-steps',str(steps),'--save-every','1']
    cmd+=['--batch-size',str(batch),*extra]
    if resume:cmd+=['--resume',str(a.work/resume)]
    result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (a.work/(out+'.log')).write_text(result.stdout)
    if result.returncode: raise RuntimeError(result.stdout)
    return result.stdout
run('full',30,1);run('chunked',7,1)
x=torch.load(a.work/'full/checkpoint-1/training.pth',weights_only=True)
y=torch.load(a.work/'chunked/checkpoint-1/training.pth',weights_only=True)
for k in x:
    if k.endswith('adam_m'):
        torch.testing.assert_close(x[k],y[k],atol=3e-5,rtol=.03)
# Full model autograd checks all six targets, both layers and v_first.
from miss_reference import forward
checkpoint=torch.load(a.work/'full/checkpoint-1/training.pth',weights_only=True)
adapters={k.removesuffix('.master'):torch.zeros_like(v,requires_grad=True)
          for k,v in checkpoint.items() if k.endswith('.master')}
tokens=torch.tensor([ord(c)+1 for c in 'abcabcabcabcabcabcabcabcabcabcabc'][:31])
logits=forward(w,adapters,tokens[:-1],8)
loss=torch.nn.functional.cross_entropy(logits,tokens[1:]);loss.backward()
for name,d in adapters.items():
    actual=checkpoint[name+'.adam_m']/.1
    torch.testing.assert_close(actual,d.grad,atol=1e-4,rtol=.01)
    assert (actual-d.grad).norm() / d.grad.norm().clamp_min(1e-12) < .005
print('Full model autograd loss:',loss.item())
run('batch',7,1,batch=2)
run('shared-tape',7,1,extra=['--wkv_tape'])
chunked=torch.load(a.work/'chunked/checkpoint-1/training.pth',weights_only=True)
for run_name in ['batch','shared-tape']:
    other=torch.load(a.work/run_name/'checkpoint-1/training.pth',weights_only=True)
    for key in chunked:
        if key.endswith('.adam_m'):
            torch.testing.assert_close(chunked[key],other[key],atol=3e-5,rtol=.03)
run('continuous',7,4);run('partial',7,2);run('resumed',7,4,'partial/checkpoint-2')
x=torch.load(a.work/'continuous/checkpoint-4/training.pth',weights_only=True)
y=torch.load(a.work/'resumed/checkpoint-4/training.pth',weights_only=True)
for k in x: torch.testing.assert_close(x[k],y[k],atol=0,rtol=0)
x=torch.load(a.work/'continuous/adapter-final.pth',weights_only=True)
y=torch.load(a.work/'resumed/adapter-final.pth',weights_only=True)
for k in x: torch.testing.assert_close(x[k],y[k],atol=0,rtol=0)
log=run('learn',7,16)
import re
losses=[float(x) for x in re.findall(r'loss=([0-9.]+)',log)]
assert losses[-1]<losses[0],losses
print('MiSS acceptance passed; loss',losses[0],'->',losses[-1])

# Dynamic runtime must reuse the same adapter semantics for all base formats.
for fmt in ['fp16','w8a16','w4a16']:
    model=a.work/'tiny.pth'
    if fmt!='fp16':
        model=a.work/('tiny.'+fmt)
        with (a.work/('quant-'+fmt+'.log')).open('w') as log:
            subprocess.run([a.build/'rwkv_quantize','--format',fmt,a.work/'tiny.pth',model],check=True,stdout=log)
    output=a.work/('logits-'+fmt+'.pth')
    with (a.work/('runtime-'+fmt+'.log')).open('w') as log:
        subprocess.run([a.build/'test/rwkv_miss_model_test',model,a.work/'learn/adapter-final.pth',output],check=True,stdout=log)
    if fmt=='fp16':
        adapter=torch.load(a.work/'learn/adapter-final.pth',weights_only=True)
        adapter={k:v.float() for k,v in adapter.items()}
        ref=forward(w,adapter,torch.tensor([98,99,100,98,99,100,98,99,100]),8)
        actual=torch.load(output,weights_only=True)['logits']
        torch.testing.assert_close(actual,ref[[5,8]],atol=.005,rtol=.003)
# Initial states remain frozen and use the PTH [H,V,K] -> runtime [H,K,V] ABI.
initial={f'blocks.{i}.att.time_state':torch.randn(1,64,64)*.001 for i in range(L)}
torch.save(initial,a.work/'initial.pth')
run('initial-state',7,1,extra=['--state',str(a.work/'initial.pth')])
state_checkpoint=torch.load(a.work/'initial-state/checkpoint-1/training.pth',weights_only=True)
expected_state=torch.cat([initial[k].transpose(-1,-2).reshape(-1) for k in sorted(initial)])
torch.testing.assert_close(state_checkpoint['initial_state'],expected_state,atol=0,rtol=0)
print('Dynamic FP16/W8A16/W4A16 model acceptance passed')
