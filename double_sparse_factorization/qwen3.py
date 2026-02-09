"""
Qwen3 Double Sparse Pruning - transformers 4.51.x 호환
기존 llama.py를 참고하여 Qwen3 모델용으로 새로 작성
"""
import time
import torch
import torch.nn as nn
from doublesparse import *
from modelutils import *

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def log_time(message):
    """Print message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")


def get_qwen3(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    log_time(f"Loading model from: {model}")
    model = Qwen3ForCausalLM.from_pretrained(model, 
                                             torch_dtype="auto",
                                             device_map="cuda:0",
                                             low_cpu_mem_usage=False)
    # model.seqlen = model.config.max_position_embeddings
    model.seqlen = 4096
    log_time(f"Model loaded. Total layers: {len(model.model.layers)}")
    return model


@torch.no_grad()
def qwen3_sequential(model, dataloader, dev):
    log_time("Starting Qwen3 pruning...")

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

    log_time("Catching first layer inputs...")
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = getattr(module, "attention_type", "full_attention")

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

    # Qwen3: position_embeddings, position_ids, cache_position 필요
    # 캐시된 값이 없으면 직접 계산 (첫 배치가 1개 샘플인 경우)
    log_time("Computing position embeddings...")
    if cache["position_embeddings"] is None or cache["position_ids"] is None:
        position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids)
        cache_position = torch.arange(model.seqlen, device=dev)
    else:
        position_embeddings = cache["position_embeddings"]
        position_ids = cache["position_ids"]
        cache_position = cache["cache_position"]

    # attention_mask가 None이면 causal mask 생성 (transformers 4.51.x)
    if attention_mask is None:
        log_time("Creating causal attention mask...")
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
            causal_masks = create_causal_mask(**mask_kwargs)
            if isinstance(causal_masks, dict):
                attention_mask = causal_masks.get("full_attention", list(causal_masks.values())[0])
            else:
                attention_mask = causal_masks
        except (ImportError, TypeError):
            # fallback: 간단한 causal mask
            attention_mask = torch.triu(
                torch.ones(model.seqlen, model.seqlen, device=dev, dtype=dtype) * float("-inf"),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

    if args.fix_mask:
        log_time("Creating fixed masks...")
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

    log_time("Ready. Starting layer-by-layer pruning...")
    quantizers = {}
    for i in range(len(layers)):
        log_time(f"Processing layer {i}/{len(layers)-1}")
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

            log_time(f"  Layer {i}: Running forward pass for {args.nsamples} samples...")
            for j in range(args.nsamples):
                # Qwen3: layer가 hidden_states만 반환 (tuple 아님)
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                }
                # position_embeddings가 배치별로 다를 수 있음 - j번째 샘플용
                if position_embeddings[0].shape[0] > 1:
                    pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                    layer_kwargs["position_embeddings"] = pe
                out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                outs[j] = out if not isinstance(out, tuple) else out[0]

            for h in handles:
                h.remove()
            del outs

            log_time(f"  Layer {i}: Pruning modules...")
            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                gpts[name].fasterprune(sparsity)
                gpts[name].free()

        log_time(f"  Layer {i}: Running forward pass after pruning...")
        outs = torch.zeros_like(inps)
        for j in range(args.nsamples):
            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }
            if position_embeddings[0].shape[0] > 1:
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
    log_time("Pruning complete!")
    return quantizers


@torch.no_grad()
def qwen3_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    log_time(f"Starting evaluation on {dataset}...")

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

    log_time("Catching first layer inputs for evaluation...")
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = getattr(module, "attention_type", "full_attention")

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
        log_time("Creating causal attention mask for evaluation...")
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
            causal_masks = create_causal_mask(**mask_kwargs)
            if isinstance(causal_masks, dict):
                attention_mask = causal_masks.get("full_attention", list(causal_masks.values())[0])
            else:
                attention_mask = causal_masks
        except (ImportError, TypeError):
            attention_mask = torch.triu(
                torch.ones(model.seqlen, model.seqlen, device=dev, dtype=dtype) * float("-inf"),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

    if position_embeddings is None:
        log_time("Computing position embeddings for evaluation...")
        position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids)
        cache_position = torch.arange(model.seqlen, device=dev)

    log_time(f"Evaluating {len(layers)} layers...")
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

    log_time("Computing perplexity...")
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
    log_time(f"Evaluation complete for {dataset}")


if __name__ == "__main__":
    import argparse
    from datautils import *

    log_time("=" * 60)
    log_time("Starting Qwen3 Double Sparse Pruning Script")
    log_time("=" * 60)

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Qwen3 model to load (e.g., Qwen/Qwen3-8B)")
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

    log_time(f"Arguments: {args}")
    log_time(f"Model: {args.model}")
    log_time(f"Dataset: {args.dataset}")
    log_time(f"Sparsity: {args.sparsity}")

    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_qwen3(args.model)
    model.eval()

    log_time("Loading calibration data...")
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    log_time(f"Calibration data loaded: {args.nsamples} samples")

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        qwen3_sequential(model, dataloader, DEV)
        log_time(f"Pruning completed in {time.time() - tick:.2f} seconds")
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "down_proj" in n:
                break
        print(time.time() - tick)
    else:
        log_time("Skipping pruning (sparsity=0 or GMP mode)")

    for dataset in ["wikitext2"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        qwen3_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        log_time(f"Saving model to: {args.save}")
        model.save_pretrained(args.save)
        log_time(f"Model saved successfully to: {args.save}")

    log_time("=" * 60)
    log_time("Script completed successfully!")
    log_time("=" * 60)
