"""Small-memory SDPA correctness screen explicitly allowed to share a GPU."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2')
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.01)
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    backends={'auto':None,'flash':SDPBackend.FLASH_ATTENTION,'efficient':SDPBackend.EFFICIENT_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}
    from contextlib import nullcontext
    for shape_index,(nq,nk,mask_kind) in enumerate([(512,512,'none'),(512,256,'bool'),(512,256,'additive')]):
        torch.manual_seed(12000+shape_index)
        tensors=[torch.randn(1,32,n,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for n in [nq,nk,nk]]
        mask=None
        if mask_kind!='none':
            valid=torch.arange(nk,device='cuda')<nk-17
            mask=valid if mask_kind=='bool' else torch.zeros(nk,device='cuda',dtype=torch.bfloat16).masked_fill(~valid,float('-inf'))
            mask=mask.view(1,1,1,nk)
        upstream=torch.randn_like(tensors[0])
        def run(backend):
            with sdpa_kernel(backend) if backend is not None else nullcontext():
                out=torch.nn.functional.scaled_dot_product_attention(*tensors,attn_mask=mask)
                grads=torch.autograd.grad(out,tensors,upstream)
            return [out.detach().float().cpu(),*[g.detach().float().cpu() for g in grads]]
        reference=run(SDPBackend.MATH)
        for name,backend in backends.items():
            row=dict(backend=name,query_tokens=nq,key_tokens=nk,mask=mask_kind)
            try:
                torch.cuda.reset_peak_memory_stats();values=run(backend)
                errors=[float((x-y).norm()/y.norm().clamp_min(1e-12)) for x,y in zip(values,reference)]
                finite=all(bool(torch.isfinite(x).all()) for x in values)
                row.update(status='supported',relative_l2=dict(zip(['output','dq','dk','dv'],errors)),
                           screening_passed=finite and errors[0]<.02 and max(errors[1:])<.03,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            except (RuntimeError,NotImplementedError,IndexError) as e:
                row.update(status='unsupported_or_failed',error=str(e)[:1600]);torch.cuda.empty_cache()
            rows.append(row)
        del tensors,reference,upstream
    result=dict(gpu=a.gpu,time=time.time(),torch=torch.__version__,shared_gpu=True,
        purpose='Compatibility and BF16 output/gradient screening only; no uncontended speed claim.',
        max_allocator_fraction=.01,rows=rows)
    (a.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
