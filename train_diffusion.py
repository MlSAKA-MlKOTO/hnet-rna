# Ad-hoc ugly training script. Do not use for anything serious.


# argparse first to make --help ASAP. please do not isort
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-ndt', '--num-diffusion-timesteps', type=int, default=32)
ap.add_argument('-c', '--config', type=str, default='./configs/hnet_2stage_small.json')
ap.add_argument('-p', '--pt-ckpt', type=str, default=None, help='e.g. ./hnet_1stage_L.pt')
ap.add_argument('-N', '--n-compression', type=str, default='1-3-9', help='''
compression depth to target with L_ratio. this is a bit different from the paper's notation;
n_compression = [1,3,9] -> N = [3/1, 9/3]
''')
ap.add_argument('-e', '--early-exit', choices=['generate', 'dumpparams'], help='''
Early-exit helpers. Use this to quickly test something && exit the script thereafter.
  generate: attempt generation via loaded ckpt.
  dumpparams: show model & all params
''')
ap.add_argument('-l', '--logger', default='local', choices=['wandb', 'local', 'neptune'])
ap.add_argument('-C', '--compile', choices=['block', 'eager'], help='attempt torch compile')
ap.add_argument('-o', '--optim', default='adamw', choices=['adamw', 'sgd'])
ap.add_argument('--lr', type=float, default=3e-4, help='adamw learning rate')
ap.add_argument('--mbs', type=int, default=1<<10, help='maximum microbatchsize (tokens per gpu)')
ap.add_argument('--steps', type=int, default=1<<10, help='total train steps')
ap.add_argument('--save-dir', type=str, help='overwriting output path to save checkpoints')
ap.add_argument('--train-data-dir', type=str, default='./datasets/train', help='directory with finetune data')
# n.b. about 76k steps required for 16ktok * 8gpu to reach 10Btok
args = ap.parse_args()





import os
import sys
import time
import functools
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from typing import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor as TT, distributed as dist
from torch.distributed import device_mesh as tdm, fsdp, checkpoint as dcp
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.checkpoint import state_dict as dcps

from omegaconf import ListConfig
from termcolor import colored

from hnet_bidirection import HNetLM, NJT, HNetConfig
from fineweb import seqlen_sorted_fineweb
from comparison import generate, HNetLM as HNetLMInference, yield_utf8_chunks
from tokenizer import RnaTokenizer
import wandb


###
### distributed 
def parent_codeline():
    parent = __import__("inspect").stack()[2]
    return f'{parent.filename}:{parent.lineno} -> {parent.code_context[0].strip()}'
def leave(): (dist.destroy_process_group() if dist.is_initialized() else None),exit() # noqa
def distdbg(): # `for f in distdbg():f()`
    if dist.get_rank() == 0:
        print('[distdbg]', parent_codeline())
        yield __import__('pdbpp').set_trace
    yield dist.barrier
def printflock(*args, fcntl=__import__('fcntl'), builtins=__import__('builtins'), **kwargs):
    r = dist.get_rank() if dist.is_initialized() else 0
    __import__("time").sleep(r*0.05) 
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try: builtins.print(f'[{r}]', *args, **kwargs)
        finally: fcntl.flock(fh, fcntl.LOCK_UN)
    if dist.is_initialized(): dist.barrier()
def pr0(*a, **k): 0 if dist.get_rank() else print(*a,**k)
def dist_equal(t: TT, g: dist.ProcessGroup):
    x = torch.zeros(g.size(), *t.shape, device=t.device, dtype=t.dtype)
    dist.all_gather_into_tensor(x, t[None], group=g)
    return all(torch.equal(x[i], t) for i in range(g.size()))
def assert_rng_equal(g: dist.ProcessGroup):
    if not dist_equal(torch.cuda.get_rng_state(), g):
        printflock(torch.cuda.get_rng_state())
        printflock('unequal! exiting at ' + parent_codeline())
        leave()
@contextmanager
def summon_full_params(model: FSDPModule):
    handles = [
        m.unshard(async_op=True)
        for m in reversed(list(model.modules()))
        if isinstance(m, FSDPModule)
    ]
    for h in handles: h.wait() if h is not None else 0

    yield

    for m in reversed(list(model.modules())):
        if isinstance(m, FSDPModule): m.reshard()


###
### tokenizer / configs
n_compression = [int(s) for s in args.n_compression.split('-')]
t = RnaTokenizer()
c = HNetConfig.load_config(args.config, N_compress=n_compression)
def load_ckpt(m: nn.Module, path: str | None):
    if path is None: return # do not do anything
    with torch.serialization.safe_globals([ListConfig]):
        d = torch.load(path, mmap=True, weights_only=False)
    m.load_state_dict(d)
def decode_sync(g): return t.decode([t for t,_ in g])

###
### Memory-free training module -> inference module copy



###
### model
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
torch.manual_seed(0)
with torch.device('cuda'): m = HNetLM(c).eval() # <-- always fp32 cuda weights
print('params:', sum(p.numel() for p in m.parameters()) / 1_000_000_000, 'B')
if args.compile == 'block': m.backbone.block_compile()


###
### dist init
ws = int(os.environ.get("WORLD_SIZE", 0))
assert ws, "Always run script with torchrun, even with only 1GPU."
dist.init_process_group("cpu:gloo,cuda:nccl")
mesh = tdm.init_device_mesh('cuda', (ws,), mesh_dim_names=('dp',))
r = dist.get_rank()

def apply_fsdp(m: HNetLM, dp_mesh: tdm.DeviceMesh):
    assert dp_mesh.ndim == 1

    # prepare fsdp helpers
    kw = dict( # default: BF16, ZeRO2, 1D mesh
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        reshard_after_forward=False,
        mesh=dp_mesh,
    )
    def shard_isotropic(iso):
        for l in iso.layers: fsdp.fully_shard(l, **kw)

    # recurse to collect all hnets
    hnets = [m.backbone]
    for s in range(c.S): hnets.append(hnets[-1].main_network)

    ## Sharding
    # special case: the lastmost hnet has a .main_network as Isotropic and nothing else.
    shard_isotropic(hnets.pop().main_network)

    # in general, shard the following:
    for hnet in hnets[::-1]:
        shard_isotropic(hnet.encoder)
        shard_isotropic(hnet.decoder)
        # NOTE: you could optimize this by packing routing & residual into 1module which fsdp fetches.
        # but it would fork the module tree away from original hnet, which I don't accept
        fsdp.fully_shard(hnet.routing_module, **kw)
        # I really do not believe this is necessary, but let's match the authors.
        fsdp.fully_shard(hnet.residual_proj, **kw|{'mp_policy':fsdp.MixedPrecisionPolicy(param_dtype=torch.float32)})

    fsdp.fully_shard(m, **kw) # top-level: .embeddings .lm_head

if ws==1:
    # Since FSDP breaks in various places for 1gpu, we can either AMP or hard-cast.
    # It is impossible to make torch.mm(..., out=...) work with AMP, so I hard-cast.
    m = m.bfloat16()
    # Neither produce the same numerical behavior as FSDP, so try to only use 1gpu for testing.
else: apply_fsdp(m,mesh)

if args.early_exit == 'dumpparams':
    pr0(m)
    for n,p in m.named_parameters(): pr0(n,p)
    leave()

###
### optim/lrs
def lr_modulation(n_gpt: float=4.6): # https://arxiv.org/pdf/2507.07955#page=35
    n_prod_ratio = n_compression[::-1]
    d_ratio = [c.d_model[-1]/d for d in c.d_model]
    return [(4.5 * n_frac * d_frac)**.5 for n_frac,d_frac in zip(n_prod_ratio, d_ratio)]
def split_params_by_hierachy(m: HNetLM) -> list[list[nn.Parameter]]:
    # for each param, count the number of times ".main_network" appears in it.
    d = defaultdict(list)
    for n,p in m.named_parameters(): d[n.count('main_network')].append(p)
    # special-case innermost hnet which has redundant .main_network
    max_depth = max(d.keys())
    assert 1 == len(d[max_depth-1]), f"expected single .pad_dimension at {max_depth-1}"
    d[max_depth-1] += d.pop(max_depth)

    return [d[k] for k in range(len(d))]
lambda_s = lr_modulation()
param_groups = [
    dict(params=plist,lr=args.lr*λˢ)
    for (plist,λˢ) in zip(split_params_by_hierachy(m), lambda_s)
]

def wsd(step: int, end: int):
    pct = step/end
    return pct*10 if pct < .1 else (
        1 if pct < .9 else (1-pct)*10
    )
opt_cls = functools.partial(torch.optim.AdamW, betas=(0.9,0.95), weight_decay=.01) if args.optim == 'adamw' else torch.optim.SGD
o = opt_cls(param_groups, lr=args.lr)
get_lr_mult = functools.partial(wsd, end=args.steps)
lrs = torch.optim.lr_scheduler.LambdaLR(o, get_lr_mult)

###
### logger
def get_logger(variant: str='local') -> Callable[[dict],None]:
    match variant:
        case 'wandb':
            import wandb
            wandb.init(project='hnet', config=asdict(c))
            return wandb.log
        case _: return print
log = get_logger(args.logger) if r==0 else lambda *a,**k:0

###
### checkpointing
def save_ckpt(step: int):
    save_dir = (Path(args.save_dir)/f'{step}')
    pr0(f'saving checkpoint to {save_dir}')
    save_dir.mkdir(exist_ok=True, parents=True)
    ckpt_m,ckpt_o = dcps.get_state_dict(m,o)
    dcp.save(dict(model=ckpt_m, optim=ckpt_o), checkpoint_id=save_dir)


def get_submod(m: nn.Module, k: str) -> tuple[nn.Module, str]:
    if '.' not in k: return m,k
    l,r = k.split('.', 1)
    return get_submod(getattr(m, l), r)

@contextmanager
def obscure_torch_wrapper_modules(m: nn.Module, *, names: list[str] = ["_fsdp_wrapped_module", "_orig_mod", "_checkpoint_wrapped_module"]):
    restore = []
    for n,c in m.named_modules():
        potential_inner = [getattr(c,k,None) for k in names]
        inner = next((p for p in potential_inner if p is not None), None)
        if inner is not None:
            parent, tail = get_submod(m,n)
            restore.append([parent, tail, c, inner])

    # m.{n} == parent.{tail} -> child; child.{k} -> inner
    for parent, tail, child, inner in restore:
        parent.register_module(tail, inner)
    yield
    for parent, tail, child, inner in restore:
        parent.register_module(tail, child)


###
### training
def cat_jagged_dim0(t0,t1) -> TT:
    values=torch.cat([t0.values(),t1.values()], dim=0)
    offsets = torch.cat([t0.offsets(), t1.offsets()[1:]+t0.offsets()[-1]])
    return torch.nested.nested_tensor_from_jagged(values, offsets)
def q_sample_coupled(x_0, t1, t2, maskable_mask,mask_id=t.mask_idx):
    t1_eq_t2_mask = t1 == t2
    t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

    # sample t1
    u = torch.rand_like(x_0, dtype=torch.float)
    t1_mask = (
        u < (t1 / args.num_diffusion_timesteps)[:, None]
    ) & maskable_mask
    x_t1 = x_0.masked_fill(t1_mask, mask_id)

    # sample t2
    u = torch.rand_like(x_0, dtype=torch.float)
    t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
    u = torch.rand_like(x_0, dtype=torch.float)
    for i in range(len(t1_eq_t2_mask)):
        if t1_eq_t2_mask[i]:
            t2_mask[i]=(
        u[i] < (t1[i] / args.num_diffusion_timesteps)[None]
    ) & (maskable_mask[i])
    # u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
    # t2_mask[t1_eq_t2_mask] = (
    #     u < (t1[t1_eq_t2_mask] / args.num_diffusion_timesteps)[:, None]
    # ) & (maskable_mask[t1_eq_t2_mask])
    x_t2 = x_0.masked_fill(t2_mask, mask_id)

    return {
        "x_t": cat_jagged_dim0(x_t1, x_t2),
        "t": torch.cat([t1, t2]),
        "mask_mask": cat_jagged_dim0(t1_mask, t2_mask),
    }

def get_non_special_symbol_mask(output_tokens, partial_masks=None,tokenizer=t):
    non_special_sym_mask = (
        output_tokens.ne(tokenizer.pad_idx)
        & output_tokens.ne(tokenizer.bos_idx)
        & output_tokens.ne(tokenizer.eos_idx)
    )
    if partial_masks is not None:
        non_special_sym_mask &= ~partial_masks
    return non_special_sym_mask

def calc_metrics(logits: TT, labels: TT, weight: TT, loss_mask: TT, numel: TT, *, ln2=torch.tensor(2,device='cuda').log()):
    ce= F.cross_entropy(logits.float(), labels, reduction='none')
    ce_sum = ce.sum().clone().detach() # local ce = sum(local ce * local weight)
    ce *= (loss_mask * weight).values()

    loss = ce.sum() / loss_mask.sum() # local loss = (local ce / local mask)

    dist.all_reduce(ce_sum)
    dist.all_reduce(numel)

    return loss, ce_sum/ln2/numel

def train_step(iids, lbls=None, *, alpha=0.03, weighting='constant'):
    t1, t2 = torch.randint(
        1,
        args.num_diffusion_timesteps + 1,
        (2 * iids.size(0),),
        device=iids.device,
    ).chunk(2)
    x_t, cur_time_step, loss_mask = list(
        q_sample_coupled(
            iids,
            t1,
            t2,
            maskable_mask=get_non_special_symbol_mask(iids),
        ).values()
    )
    if lbls is None: lbls = cat_jagged_dim0(iids,iids)
    else:
        lbls=cat_jagged_dim0(lbls,lbls)
    numel = torch.tensor(lbls.numel(), dtype=torch.long).to('cuda', non_blocking=True)
    logits, loss_rt, comp_ratios = m(x_t)
    num_timesteps = args.num_diffusion_timesteps
    weight = {
        "linear": (
            num_timesteps - (cur_time_step - 1)
        ),  # num_timesteps * (1 - (t-1)/num_timesteps)
        "constant": num_timesteps * torch.ones_like(cur_time_step),
    }[weighting][:, None].float() / num_timesteps

    # loss_ce, bpb = calc_metrics(logits.values(), lbls.values(), numel)
    loss_diff,bpb = calc_metrics(logits.values(), lbls.values(), weight, loss_mask, numel)

    loss = loss_diff+alpha*loss_rt
    loss.backward()

    o.step()
    o.zero_grad()
    lrs.step()

    metrics = torch.stack([loss, loss_diff, loss_rt, bpb])
    return metrics.tolist() + [comp_ratios] # <-- cpu sync

# def seq2args(ls: list[bytes]) -> tuple[TT,TT]:
#     samples = [torch.tensor(bytearray(b),device='cuda',dtype=torch.int) for b in ls]
#     iids = NJT([s[:-1] for s in samples])
#     lbls = NJT([s[1: ] for s in samples]).long()
#     return iids, lbls

def rna_dataloader(file_path: str, rank: int, wsize: int, msl: int = 1 << 15):
    """
    RNA from .txt
    follow rank and wsize for split
    """
    lines=[]
    for f_single in os.listdir(file_path):
        if not f_single.endswith('.txt'):
            continue
        with open(os.path.join(file_path,f_single)) as f:
            lines.extend([t.strip() for t in f.readlines()])

    lines_for_this_rank = lines[rank::wsize]
    seqs, plen = [], 0
    while True:
        for line in lines_for_this_rank:
            rna_seq = line.strip()
            if not rna_seq:
                continue

            seq_len = len(rna_seq) + 2  # +2 means adding BOS and EOS token
            if seq_len > msl:
                rna_seq = rna_seq[:msl-2]  # truncate to max sequence length
                seq_len = msl
            if plen + seq_len > msl:
                yield seqs
                seqs, plen = [rna_seq], seq_len
            else:
                seqs.append(rna_seq)
                plen += seq_len
        
        # yield left seqs at the end of epoch 
        if seqs:
            yield seqs
            seqs, plen = [], 0

    
def seq2args_rna(ls: list[str],tokenizer=t) -> tuple[TT, TT]:
    """
    transform RNA sequences into input and label tensors.
    """
    # add <bos> token and tokenization
    tokenized_seqs = tokenizer.encode(ls, add_special_tokens=True)
    
    samples = [torch.tensor(seq['input_ids'], device='cuda', dtype=torch.int) for seq in tokenized_seqs]
    
    iids = NJT(samples)
    lbls = NJT(samples).long()
    return iids, lbls


# add your own if needed; there are no other gpus in my observable reality
# gpuflops = {
#     'NVIDIA GeForce RTX 3090':71e12,
#     'NVIDIA GeForce RTX 4090':165.15e12,
#     'NVIDIA B200': 2250e12,
#     'NVIDIA GeForce RTX 4060 Laptop GPU': 22.5e12,
# }[torch.cuda.get_device_name()]


assert_rng_equal(dist.group.WORLD) # <-- assumes node of homogenous GPUs
if __name__ == '__main__':
    pr0('start training')
    for step,batch in enumerate(rna_dataloader(args.train_data_dir,r, ws, args.mbs)):
        # train step
        t_step = time.perf_counter()
        iids, lbls = seq2args_rna(batch)
        loss, loss_diff, loss_rt, bpb, comp_ratios = train_step(iids, lbls)

        # calc mfu (underestimate since we don't know avg seqlen)
        batch_flops = m.flops(iids.values().shape[0], iids._max_seqlen)
        t_delta = time.perf_counter() - t_step
        # mfu = batch_flops / (gpuflops*t_delta)

        # Log (every 10steps)
        lr_list = [mult*args.lr*get_lr_mult(step) for mult in lambda_s]
        log_step = log if step % 10 == 0 else lambda d:0
        log_step({
            'step': step,
            'batch_flops (overestimate)': batch_flops,
            # 'mfu (underestimate)': mfu,
            'bpb': bpb,
            'loss/diff': loss_diff,
            'loss/ratio': loss_rt,
            'loss/total': loss,
            **{f'Compression L{i+1}/L{i}':ratio for i,ratio in enumerate(comp_ratios)},
            **{f'lr/{i}':lr for i,lr in enumerate(lr_list)},
        })

        # # Try sampling (every 100)
        # if step % 100 == 0:
        #     pr0('generating...')
        #     with summon_full_params(m) if isinstance(m, FSDPModule) else nullcontext():
        #         with obscure_torch_wrapper_modules(m):
        #             m_inf = create_inference_model_clone(m,c)
        #         p = 'AUCGNCCCC'
        #         with torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
        #             try: res1 = ''.join(
        #                 colored(c, 'white' if i%2 else 'black', 'on_black' if i%2 else 'on_white')
        #                 for i,c in enumerate(yield_utf8_chunks(generate(m_inf, p, 512,tokenizer=t),tokenizer=t))
        #             )
        #             except UnicodeDecodeError: res1 = colored('[failed to decode UTF-8]', 'red')
        #     pr0(colored(p, attrs=['underline']) + res1)
        #     pr0('='*50)


        # Try saving (every 1000)
        if step % 1000 == 0:
            save_ckpt(step) if args.save_dir else pr0(f"not saving checkpoint as {args.save_dir=}")

        if step == args.steps: break

