"""
LLaMA Double Sparse Pruning - transformers 4.51.x 호환
기존 llama.py를 참고하여 transformers 4.51.2 기준으로 새로 작성
"""
import sys
import time

import torch
import torch.nn as nn

# torchvision 호환성 사전 검사 (transformers import 시 torchvision::nms 오류 방지)
def _check_torchvision():
    try:
        import torchvision  # noqa: F401
    except RuntimeError as e:
        if "torchvision::nms" in str(e):
            print("\n" + "=" * 70, file=sys.stderr)
            print("ERROR: torch / torchvision 버전 불일치 (torchvision::nms does not exist)", file=sys.stderr)
            print("", file=sys.stderr)
            print("해결 방법:", file=sys.stderr)
            print("  1. torch와 torchvision을 함께 재설치:", file=sys.stderr)
            print("     pip uninstall torch torchvision -y", file=sys.stderr)
            print("     pip install torch torchvision", file=sys.stderr)
            print("", file=sys.stderr)
            print("  2. CUDA 사용 시 (예: CUDA 12.1):", file=sys.stderr)
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
            print("", file=sys.stderr)
            print("  3. repo 내 transformers 폴더가 있다면 제거 후 pip install transformers", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)
            sys.exit(1)
        raise

_check_torchvision()

from doublesparse import *
from modelutils import *

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def _import_transformers():
    """transformers import 시 torchvision::nms 오류 방지/처리"""
    try:
        from transformers import LlamaForCausalLM
        return LlamaForCausalLM
    except RuntimeError as e:
        if "torchvision::nms" in str(e):
            print("\n" + "=" * 70, file=sys.stderr)
            print("ERROR: torch / torchvision 버전 불일치 (torchvision::nms does not exist)", file=sys.stderr)
            print("", file=sys.stderr)
            print("해결 방법:", file=sys.stderr)
            print("  1. torch와 torchvision을 함께 재설치:", file=sys.stderr)
            print("     pip uninstall torch torchvision -y", file=sys.stderr)
            print("     pip install torch torchvision", file=sys.stderr)
            print("", file=sys.stderr)
            print("  2. CUDA 사용 시 (예: CUDA 12.1):", file=sys.stderr)
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
            print("", file=sys.stderr)
            print("  3. 현재 버전 확인:", file=sys.stderr)
            print(f"     torch: {torch.__version__}", file=sys.stderr)
            try:
                import torchvision
                print(f"     torchvision: {torchvision.__version__}", file=sys.stderr)
            except Exception:
                pass
            print("=" * 70 + "\n", file=sys.stderr)
        raise


def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    LlamaForCausalLM = _import_transformers()

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting LLaMA pruning (transformers 4.51.x)...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_embeddings": None,
        "position_ids": None,
        "cache_position": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["cache_position"] = kwargs.get("cache_position")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    # transformers 4.51.x: position_embeddings, position_ids, cache_position 필요
    if cache["position_embeddings"] is None or cache["position_ids"] is None:
        position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids=position_ids)
        cache_position = torch.arange(model.seqlen, device=dev)
    else:
        position_embeddings = cache["position_embeddings"]
        position_ids = cache["position_ids"]
        cache_position = cache["cache_position"]

    # attention_mask가 None이면 causal mask 생성 (transformers 4.51.x)
    if attention_mask is None:
        try:
            try:
                from transformers.masking_utils import create_causal_mask
            except ImportError:
                from transformers.modeling_attn_mask_utils import create_causal_mask
            mask_kwargs = {
                "config": model.config,
                "input_embeds": inps[0:1],
                "attention_mask": None,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            attention_mask = create_causal_mask(**mask_kwargs)
        except (ImportError, TypeError):
            attention_mask = torch.triu(
                torch.ones(model.seqlen, model.seqlen, device=dev, dtype=dtype) * float("-inf"),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

    if args.fix_mask:
        masks = {}
        for n, p in model.named_parameters():
            if "layers" in n and "weight" in n and len(p.shape) == 2:
                shape_key = min(p.shape), max(p.shape)
                if shape_key in masks:
                    continue
                dim = shape_key[0]
                nnz = 0.1 if shape_key[0] == shape_key[1] else 0.2
                print(n, p.shape, shape_key, nnz)
                A = torch.eye(dim, device="cuda")
                Arand = torch.rand_like(A)
                Arand += A * 100
                thres = Arand.abs().flatten().sort()[0][int(A.numel() * (1 - nnz))]
                masks[shape_key] = (Arand.abs() > thres)

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue

                fixmask = None
                if args.fix_mask:
                    shape_key = min(subset[name].weight.shape), max(subset[name].weight.shape)
                    fixmask = masks[shape_key]
                gpts[name] = DoubleSparse(subset[name], nofinal=args.no_final, fixmask=fixmask)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }

            for j in range(args.nsamples):
                if position_embeddings is not None and position_embeddings[0].shape[0] > 1:
                    pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                    layer_kwargs["position_embeddings"] = pe
                out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                outs[j] = out if not isinstance(out, tuple) else out[0]

            for h in handles:
                h.remove()
            del outs

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                gpts[name].fasterprune(sparsity)
                gpts[name].free()

        outs = torch.zeros_like(inps)
        for j in range(args.nsamples):
            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }
            if position_embeddings is not None and position_embeddings[0].shape[0] > 1:
                pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                layer_kwargs["position_embeddings"] = pe
            out = layer(inps[j].unsqueeze(0), **layer_kwargs)
            outs[j] = out if not isinstance(out, tuple) else out[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_embeddings": None,
        "position_ids": None,
        "cache_position": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["cache_position"] = kwargs.get("cache_position")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]
    position_ids = cache["position_ids"]
    cache_position = cache["cache_position"]

    if attention_mask is None:
        try:
            try:
                from transformers.masking_utils import create_causal_mask
            except ImportError:
                from transformers.modeling_attn_mask_utils import create_causal_mask
            mask_kwargs = {
                "config": model.config,
                "input_embeds": inps[0:1],
                "attention_mask": None,
                "cache_position": torch.arange(model.seqlen, device=dev),
                "past_key_values": None,
                "position_ids": torch.arange(model.seqlen, device=dev).unsqueeze(0),
            }
            attention_mask = create_causal_mask(**mask_kwargs)
        except (ImportError, TypeError):
            attention_mask = torch.triu(
                torch.ones(model.seqlen, model.seqlen, device=dev, dtype=dtype) * float("-inf"),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

    if position_embeddings is None:
        position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids=position_ids)
        cache_position = torch.arange(model.seqlen, device=dev)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        layer_kwargs = {
            "attention_mask": attention_mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

        for j in range(nsamples):
            if position_embeddings is not None and position_embeddings[0].shape[0] > 1:
                pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                layer_kwargs["position_embeddings"] = pe
            out = layer(inps[j].unsqueeze(0), **layer_kwargs)
            outs[j] = out if not isinstance(out, tuple) else out[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LLaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--no-final", action="store_true", help="Do not run the finalizer."
    )
    parser.add_argument(
        "--fix-mask", action="store_true", help="Keep one mask fixed."
    )
    args = parser.parse_args()

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        llama_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "down_proj" in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
