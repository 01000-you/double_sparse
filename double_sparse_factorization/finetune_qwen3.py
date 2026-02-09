"""
Double Sparse로 프루닝된 Qwen3 모델 파인튜닝
- 마스크는 저장된 가중치에서 복원: (weight != 0) → 학습 가능, (weight == 0) → gradient 0으로 고정
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Qwen3ForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset


# 프루닝된 레이어 (qwen3_sequential과 동일)
PRUNED_LAYER_SUBSTRINGS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def get_pruned_param_names(model):
    """프루닝된 파라미터 (weight에 0이 있는 레이어)"""
    pruned = []
    for name, param in model.named_parameters():
        if param.dim() == 2 and any(s in name for s in PRUNED_LAYER_SUBSTRINGS):
            if (param == 0).any():
                pruned.append(name)
    return pruned


def mask_gradients(model, pruned_param_names):
    """weight==0 위치의 gradient를 0으로 (해당 위치는 업데이트 안 함)"""
    for name in pruned_param_names:
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            param.grad[param == 0] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="프루닝된 모델 경로 (--save로 저장한 경로)")
    parser.add_argument("--output_dir", type=str, default="./finetuned_output")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=-1, help="-1이면 전체 epoch")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print("Loading pruned model...")
    model = Qwen3ForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(device)

    pruned_param_names = get_pruned_param_names(model)
    print(f"Pruned params (gradient masking): {len(pruned_param_names)}")
    for n in pruned_param_names[:5]:
        print(f"  - {n}")
    if len(pruned_param_names) > 5:
        print(f"  ... 외 {len(pruned_param_names)-5}개")

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, args.dataset_config, split="train")

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = [[x for x in ids] for ids in out["input_ids"]]
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def collate_fn(batch):
        d = {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}
        d["labels"][d["attention_mask"] == 0] = -100
        return d

    dataloader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.num_epochs if args.max_steps <= 0 else min(args.max_steps, len(dataloader) * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = args.bf16 and device.type == "cuda"

    model.train()
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                loss.backward()
                mask_gradients(model, pruned_param_names)
                optimizer.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                mask_gradients(model, pruned_param_names)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if global_step % args.log_steps == 0:
                print(f"Step {global_step}, loss: {loss.item():.4f}")

            if global_step > 0 and global_step % args.save_steps == 0:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Saved checkpoint at step {global_step}")

            global_step += 1

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
