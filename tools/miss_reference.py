"""Small, deliberately unfused PyTorch RWKV-7 + MiSS correctness oracle."""
import torch
import torch.nn.functional as F

def forward(weights, adapters, tokens, rank, scale=1):
    q = lambda x: x.half().float()
    w = {k: q(v.float()) for k, v in weights.items()}
    C = w['emb.weight'].shape[1]
    H = C // 64
    x = q(F.layer_norm(w['emb.weight'][tokens], (C,), w['blocks.0.ln0.weight'],
                       w['blocks.0.ln0.bias'], 1e-5))
    first = None
    l = 0
    while f'blocks.{l}.ln1.weight' in w:
        prefix = f'blocks.{l}.'
        get = lambda n: w[prefix+n]
        def linear(x, name):
            base = q(x @ get(name).T)
            d = adapters.get(prefix+name+'.D')
            if d is None: return base
            padded = F.pad(x, (0, (-x.shape[-1]) % rank))
            s = padded.reshape(x.shape[0], -1, rank).sum(1)
            return q(base + scale * s @ d.T)
        def mix(x, name):
            prev = F.pad(x[:-1], (0, 0, 1, 0))
            return q(x + (prev-x)*get(name))
        xx = q(F.layer_norm(x,(C,),get('ln1.weight'),get('ln1.bias'),1e-5))
        xr,xw,xk,xv,xa,xg = [mix(xx,'att.x_'+n) for n in 'rwkvag']
        r=linear(xr,'att.receptance.weight')
        k=linear(xk,'att.key.weight')
        v=linear(xv,'att.value.weight')
        raw_w=q(q(q(torch.tanh(q(xw@get('att.w1'))))@get('att.w2'))+get('att.w0'))
        a12=q(q(xa@get('att.a1'))@get('att.a2'))
        alpha=torch.sigmoid(get('att.a0')+a12)
        g=q(q(torch.sigmoid(q(xg@get('att.g1'))))@get('att.g2'))
        if l==0: first=v
        else:
            gate=torch.sigmoid(get('att.v0')+q(q(xv@get('att.v1'))@get('att.v2')))
            v=q(v+(first-v)*gate)
        u=(k*get('att.k_k')).reshape(-1,H,64)
        kk=(u/u.norm(dim=-1,keepdim=True).clamp_min(1e-12)).reshape(-1,C)
        new_k=q(k*(1-get('att.k_a')+alpha*get('att.k_a')))
        neg_kk=q(-kk);kka=q(kk*alpha)
        state=torch.zeros(H,64,64)
        ys=[]
        for t in range(len(tokens)):
            decay=torch.exp(-torch.exp(torch.tensor(-.5))*torch.sigmoid(raw_w[t])).reshape(H,64)
            sa=torch.einsum('hkv,hk->hv',state,neg_kk[t].reshape(H,64))
            state=state*decay[:,:,None]+kka[t].reshape(H,64,1)*sa[:,None,:]+new_k[t].reshape(H,64,1)*v[t].reshape(H,1,64)
            ys.append(q(torch.einsum('hkv,hk->hv',state,r[t].reshape(H,64))).flatten())
        y=torch.stack(ys).reshape(-1,H,64)
        gn=(y-y.mean(-1,keepdim=True))*torch.rsqrt(y.var(-1,unbiased=False,keepdim=True)+64e-5)
        gn=gn.reshape(-1,C)*get('att.ln_x.weight')+get('att.ln_x.bias')
        residual=(r*new_k*get('att.r_k').flatten()).reshape(-1,H,64).sum(-1,keepdim=True)
        post=q((gn+(residual*v.reshape(-1,H,64)).reshape(-1,C))*g)
        x=q(x+linear(post,'att.output.weight'))
        xx=q(F.layer_norm(x,(C,),get('ln2.weight'),get('ln2.bias'),1e-5))
        mixed=mix(xx,'ffn.x_k')
        hidden=linear(mixed,'ffn.key.weight')
        act=q(F.relu(hidden).square())
        x=q(x+linear(act,'ffn.value.weight'))
        l+=1
    x=q(F.layer_norm(x,(C,),w['ln_out.weight'],w['ln_out.bias'],1e-5))
    return q(x@w['head.weight'].T)
