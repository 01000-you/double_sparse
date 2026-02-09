"""
Qwen3 전용: 2SSP + Double Sparse Factorization 결합 프루닝

프로세스 (사용자 의도):
1. W를 AB로 분해 (초기 분해)
2. 2SSP 알고리즘으로 AB 사이 어떤 채널을 reduction할지 결정
3. 채널 마스크 고정 후 ADMM으로 최적화

MLP + Attention 모두 적용 (2SSP와 동일)
"""
import time
import torch
import torch.nn as nn
from doublesparse_2ssp import (
    DoubleSparse2SSP,
    initial_factorize,
    get_channel_keep_mask,
    compute_num_keep_from_target_compression,
)
from modelutils import find_layers, DEV

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def log_time(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")


def get_qwen3(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    log_time(f"Loading model from: {model}")
    model = Qwen3ForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="cuda:0",
        low_cpu_mem_usage=False,
    )
    model.seqlen = 4096
    log_time(f"Model loaded. Total layers: {len(model.model.layers)}")
    return model


def _setup_position_embeddings(model, inps, dev):
    """position_embeddings, attention_mask 등 설정"""
    position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(inps[0:1], position_ids)
    cache_position = torch.arange(model.seqlen, device=dev)
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
    return attention_mask, position_embeddings, position_ids, cache_position


def _compute_compression_stats(model, num_layers, hidden_size, intermediate_size,
                               num_keep_mlp, num_keep_attn, args):
    """W 총 파라미터 vs A+B 저장 시 파라미터 (채널 마스크 적용 후)"""
    total_w = 0
    total_ab_saved = 0
    keep_ratio_mlp = num_keep_mlp / intermediate_size
    keep_ratio_attn = num_keep_attn / hidden_size

    for i in range(num_layers):
        if not (args.minlayer <= i < args.maxlayer):
            continue
        layer = model.model.layers[i]
        full = find_layers(layer)

        # MLP: up, gate (channel=intermediate), down (channel=intermediate)
        for name in ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]:
            if name not in full:
                continue
            if args.prune_only and args.prune_only not in name:
                continue
            w = full[name].weight
            out, in_dim = w.shape[0], w.shape[1]
            total_w += out * in_dim
            if "down" in name:
                ab = out * in_dim + in_dim * in_dim
            else:
                ab = out * out + out * in_dim
            total_ab_saved += int(ab * keep_ratio_mlp)

        # Attn: q,k,v,o_proj (channel=hidden)
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
            if name not in full:
                continue
            if args.prune_only and args.prune_only not in name:
                continue
            w = full[name].weight
            h = w.shape[0]
            total_w += h * h
            total_ab_saved += int(2 * h * h * keep_ratio_attn)

    return total_w, total_ab_saved


def _get_channel_importance_2ssp(model, calibration_input_ids, num_layers,
                                  intermediate_size, hidden_size, seqlen, device):
    """
    2SSP: MLP hidden state L2 norm + Attention output L2 norm
    Returns: (mlp_norms, attn_norms)
    mlp_norms[layer_i] = [intermediate_size]
    attn_norms[layer_i] = [hidden_size]  (q,k,v,o_proj 출력 norm)
    """
    mlp_norms = {}
    attn_norms = {}

    def make_hook(store_dict, key):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach().reshape(-1, out.size(-1))
            if key not in store_dict:
                store_dict[key] = []
            store_dict[key].append(out)
        return hook

    hooks = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        layer.mlp.down_proj._2ssp_li = ("mlp", i)
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, name)
            proj._2ssp_li = ("attn", i)

    mlp_store = {}
    attn_store = {}

    def mlp_hook(module, input, output):
        h = input[0].detach().reshape(-1, input[0].size(-1))
        li = module._2ssp_li[1]
        if li not in mlp_store:
            mlp_store[li] = []
        mlp_store[li].append(h)

    def attn_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.detach().reshape(-1, out.size(-1))
        li = module._2ssp_li[1]
        if li not in attn_store:
            attn_store[li] = []
        attn_store[li].append(out)

    for i in range(num_layers):
        hooks.append(model.model.layers[i].mlp.down_proj.register_forward_hook(mlp_hook))
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(model.model.layers[i].self_attn, name)
            hooks.append(proj.register_forward_hook(attn_hook))

    model.eval()
    with torch.no_grad():
        total_len = calibration_input_ids.size(1)
        step = max(seqlen // 2, 1)
        for i in range(0, total_len - seqlen + 1, step):
            batch = calibration_input_ids[:, i : i + seqlen].to(device)
            _ = model(batch)

    for h in hooks:
        h.remove()

    for li in range(num_layers):
        if li in mlp_store:
            stacked = torch.cat(mlp_store[li], dim=0)
            mlp_norms[li] = stacked.norm(dim=0, p=2).to(device)
        else:
            mlp_norms[li] = torch.ones(intermediate_size, device=device)

        if li in attn_store:
            stacked = torch.cat(attn_store[li], dim=0)
            attn_norms[li] = stacked.norm(dim=0, p=2).to(device)
        else:
            attn_norms[li] = torch.ones(hidden_size, device=device)

    return mlp_norms, attn_norms


@torch.no_grad()
def qwen3_sequential_2ssp(model, dataloader, dev, channel_reduction_ratio=0.5, target_compression=None):
    """
    프로세스:
    1. W를 AB로 초기 분해 (채널 제약 없음)
    2. 2SSP로 AB 사이 어떤 채널 reduction할지 결정
    3. ADMM으로 채널 마스크 고정 후 최적화
    """
    log_time("Starting Qwen3 2SSP+DoubleSparse (AB분해 -> 2SSP채널선택 -> ADMM)")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    num_layers = len(layers)
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_embeddings": None,
             "position_ids": None, "cache_position": None}

    log_time("Phase 0: Catching inputs...")
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
    calibration_batches = []
    for batch in dataloader:
        try:
            calibration_batches.append(batch[0].to(dev))
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if not calibration_batches:
        raise RuntimeError("No calibration data captured")
    calibration_input_ids = torch.cat(calibration_batches, dim=1)
    if calibration_input_ids.size(0) > 1:
        calibration_input_ids = calibration_input_ids[0:1]

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_mask, position_embeddings, position_ids, cache_position = _setup_position_embeddings(
        model, inps, dev
    )

    seqlen = min(2048, model.seqlen)
    if target_compression is not None:
        num_keep_mlp, num_keep_attn = compute_num_keep_from_target_compression(
            target_compression, hidden_size, intermediate_size
        )
        log_time(f"Target compression {target_compression:.0%} -> num_keep MLP={num_keep_mlp}/{intermediate_size}, Attn={num_keep_attn}/{hidden_size}")
    else:
        num_keep_mlp = max(1, int(intermediate_size * (1 - channel_reduction_ratio)))
        num_keep_attn = max(1, int(hidden_size * (1 - channel_reduction_ratio)))

    log_time("Phase 1: Initial AB factorization (all layers)...")
    outs = torch.zeros_like(inps)
    for i in range(num_layers):
        log_time(f"  Layer {i}: Initial AB factorization...")
        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                if not (args.minlayer <= i < args.maxlayer):
                    continue
                if args.prune_only and args.prune_only not in name:
                    continue
                gpts[name] = DoubleSparse2SSP(subset[name], channel_keep_mask=None, nofinal=args.no_final)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[n].register_forward_hook(add_batch(n)) for n in subset if n in gpts]
            if not handles:
                continue

            for j in range(args.nsamples):
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                }
                if position_embeddings is not None and len(position_embeddings) >= 2 and position_embeddings[0].shape[0] > 1:
                    pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                    layer_kwargs["position_embeddings"] = pe
                out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                outs[j] = out if not isinstance(out, tuple) else out[0]

            for h in handles:
                h.remove()

            for name in subset:
                if name not in gpts:
                    continue
                W = subset[name].weight.data.clone().float()
                H = gpts[name].H
                W_ab = initial_factorize(W, H)
                subset[name].weight.data = W_ab.reshape(subset[name].weight.shape).to(subset[name].weight.dtype)
                gpts[name].free()

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    log_time("Phase 2: 2SSP channel importance (model now uses A@B)...")
    model = model.to(dev)
    mlp_norms, attn_norms = _get_channel_importance_2ssp(
        model, calibration_input_ids, num_layers,
        intermediate_size, hidden_size, seqlen, dev
    )

    channel_masks_mlp = {li: get_channel_keep_mask(mlp_norms[li], num_keep_mlp) for li in range(num_layers)}
    channel_masks_attn = {li: get_channel_keep_mask(attn_norms[li], num_keep_attn) for li in range(num_layers)}

    log_time(f"Channel reduction: MLP {intermediate_size}->{num_keep_mlp}, Attn {hidden_size}->{num_keep_attn}")

    log_time("Phase 3: ADMM with channel mask...")
    inps = torch.zeros((args.nsamples, model.seqlen, hidden_size), dtype=dtype, device=dev)
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
    for i in range(num_layers):
        log_time(f"  Layer {i}: ADMM with channel mask...")
        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                if not (args.minlayer <= i < args.maxlayer):
                    continue
                if args.prune_only and args.prune_only not in name:
                    continue

                if "mlp" in name:
                    channel_mask = channel_masks_mlp[i]
                    channel_dim = "in" if name == "mlp.down_proj" else "out"
                else:
                    channel_mask = channel_masks_attn[i]
                    channel_dim = "out"

                gpts[name] = DoubleSparse2SSP(
                    subset[name], channel_keep_mask=channel_mask,
                    nofinal=args.no_final, channel_dim=channel_dim
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[n].register_forward_hook(add_batch(n)) for n in subset if n in gpts]
            if not handles:
                continue

            for j in range(args.nsamples):
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                }
                if position_embeddings is not None and len(position_embeddings) >= 2 and position_embeddings[0].shape[0] > 1:
                    pe = (position_embeddings[0][j:j+1], position_embeddings[1][j:j+1])
                    layer_kwargs["position_embeddings"] = pe
                out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                outs[j] = out if not isinstance(out, tuple) else out[0]

            for h in handles:
                h.remove()

            for name in subset:
                if name in gpts:
                    log_time(f"    Pruning {name}...")
                    gpts[name].fasterprune()
                    gpts[name].free()

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    log_time("Pruning complete!")

    # report actual compression (W vs A+B saved)
    total_w, total_ab_saved = _compute_compression_stats(
        model, num_layers, hidden_size, intermediate_size,
        num_keep_mlp, num_keep_attn, args
    )
    if total_w > 0:
        actual_ratio = total_ab_saved / total_w
        log_time(f"[Compression] W total: {total_w:,} | A+B saved: {total_ab_saved:,} | ratio: {actual_ratio:.2%}")

    return {}


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
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_embeddings": None, "position_ids": None, "cache_position": None}

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
    if position_embeddings is None:
        position_ids = torch.arange(model.seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids)
        cache_position = torch.arange(model.seqlen, device=dev)

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer_kwargs = {
            "attention_mask": attention_mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        for j in range(nsamples):
            if position_embeddings is not None and len(position_embeddings) >= 2 and position_embeddings[0].shape[0] > 1:
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
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
    from datautils import get_loaders

    log_time("=" * 60)
    log_time("Qwen3 2SSP + Double Sparse (AB분해 -> 2SSP -> ADMM)")
    log_time("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Qwen3 model (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"], help="Calibration dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--channel-reduction", type=float, default=0.5,
                        help="Channel reduction ratio (0.5 = remove 50%% of channels)")
    parser.add_argument("--target-compression", type=float, default=None,
                        help="W 대비 A+B 저장 목표 압축률 (0.3 = 30%% 압축). 지정 시 channel-reduction 무시")
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune_only", type=str, default="",
                        help="Prune only layers containing this (empty = all)")
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--log_wandb", action="store_true")

    args = parser.parse_args()

    log_time(f"Model: {args.model}, Dataset: {args.dataset}")
    if args.target_compression is not None:
        log_time(f"Target compression: {args.target_compression:.0%} (W 대비 A+B 저장)")
    else:
        log_time(f"Channel reduction: {args.channel_reduction}")

    if args.log_wandb and has_wandb:
        wandb.init(config=args)

    model = get_qwen3(args.model)
    model.eval()

    log_time("Loading calibration data...")
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=4096
    )

    tick = time.time()
    qwen3_sequential_2ssp(
        model, dataloader, DEV,
        channel_reduction_ratio=args.channel_reduction,
        target_compression=args.target_compression,
    )
    log_time(f"Pruning completed in {time.time() - tick:.2f}s")

    for dataset in ["wikitext2"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=4096
        )
        print("Dataset:", dataset)
        qwen3_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        log_time(f"Saving model to {args.save}")
        model.save_pretrained(args.save)

    log_time("=" * 60)
    log_time("Done!")
