"""
저장된 프루닝 모델 로드 후 Perplexity 평가
- qwen3.py --save로 저장한 모델 경로를 입력
"""
import argparse
import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, AutoTokenizer

from datautils import get_loaders
from modelutils import DEV


def eval_model(model, testenc, dev, seqlen, dataset_name=""):
    """Layer-by-layer 평가 (메모리 효율)"""
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_embeddings": None, "position_ids": None, "cache_position": None}

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
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
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
                "cache_position": torch.arange(seqlen, device=dev),
                "past_key_values": None,
                "position_ids": torch.arange(seqlen, device=dev).unsqueeze(0),
            }
            causal_masks = create_causal_mask(**mask_kwargs)
            if isinstance(causal_masks, dict):
                attention_mask = causal_masks.get("full_attention", list(causal_masks.values())[0])
            else:
                attention_mask = causal_masks
        except (ImportError, TypeError):
            attention_mask = torch.triu(
                torch.ones(seqlen, seqlen, device=dev, dtype=dtype) * float("-inf"), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

    if position_embeddings is None:
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inps[0:1], position_ids)
        cache_position = torch.arange(seqlen, device=dev)

    for i in range(len(layers)):
        print(f"  Layer {i}/{len(layers)}")
        layer = layers[i].to(dev)
        layer_kwargs = {
            "attention_mask": attention_mask,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        for j in range(nsamples):
            if position_embeddings is not None and position_embeddings[0].shape[0] > 1:
                pe = (position_embeddings[0][j : j + 1], position_embeddings[1][j : j + 1])
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
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def eval_model_fast(model, testenc, dev, seqlen):
    """전체 모델 forward (메모리 충분할 때, 더 빠름)"""
    model = model.to(dev)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        nlls.append(outputs.loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description="저장된 프루닝 모델 평가")
    parser.add_argument("model_path", type=str, help="저장된 모델 경로 (--save로 저장한 경로)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4"],
        help="평가 데이터셋",
    )
    parser.add_argument("--seqlen", type=int, default=2048, help="시퀀스 길이")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="전체 모델 forward (빠름, 메모리 많이 사용)")
    args = parser.parse_args()

    dev = DEV
    print(f"Loading model from {args.model_path}...")
    model = Qwen3ForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")
    model.eval()
    model.seqlen = getattr(model.config, "max_position_embeddings", args.seqlen)

    print(f"Loading dataset: {args.dataset}...")
    _, testenc = get_loaders(
        args.dataset, nsamples=1, seed=args.seed, model=args.model_path, seqlen=args.seqlen
    )

    seqlen = min(args.seqlen, model.seqlen)
    print(f"Evaluating (seqlen={seqlen}, mode={'fast' if args.fast else 'layer-by-layer'})...")
    if args.fast:
        ppl = eval_model_fast(model, testenc, dev, seqlen)
    else:
        ppl = eval_model(model, testenc, dev, seqlen, args.dataset)
    print(f"\n{args.dataset} Perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
