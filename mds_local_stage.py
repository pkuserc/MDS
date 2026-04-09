#!/usr/bin/env python3
"""
MDS Local Stage: dialogue-level local scoring and bin-wise selection.
Usgae:

python mds_local_stage.py \
  --input_jsonl outputs/mds_global_candidates.jsonl \
  --output_jsonl outputs/mds_final_selected.jsonl \
  --output_ids_json outputs/mds_final_selected_ids.json \
  --judge_model Qwen/Qwen3-8B \
  --budget 10000 \
  --tau_form 1.0

"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch.distributed as dist
except Exception:
    dist = None


# ----------------------------
# Data helpers
# ----------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_conv_id(ex: Dict[str, Any]) -> Any:
    return ex.get("conv_id", ex.get("id", ex.get("conversation_id", None)))


def normalize_dialog(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize supported dialogue formats into:
      {
        "conv_id": ...,
        "bin_id": ... (optional),
        "dialog": [{"role": "...", "text": "..."}, ...]
      }
    """
    cid = get_conv_id(ex)
    if cid is None:
        raise ValueError("Missing conv_id/id in an example.")

    bin_id = ex.get("bin_id", ex.get("cluster_id", ex.get("bin", None)))

    turns = ex.get("dialog", None)
    if turns is None:
        turns = ex.get("messages", [])

    norm_turns: List[Dict[str, str]] = []
    for t in turns:
        role = str(t.get("role", "")).lower()
        text = t.get("text", t.get("content", ""))
        if text is None:
            text = ""
        norm_turns.append({"role": role, "text": str(text)})

    return {"conv_id": cid, "bin_id": bin_id, "dialog": norm_turns, "_raw": ex}


def extract_qa_pairs(dialog: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Extract (question, answer) pairs:
      user -> next assistant
    """
    qa: List[Tuple[str, str]] = []
    i = 0
    n = len(dialog)
    while i < n:
        if dialog[i].get("role") == "user":
            q = dialog[i].get("text", "")
            j = i + 1
            a = None
            while j < n:
                if dialog[j].get("role") == "assistant":
                    a = dialog[j].get("text", "")
                    break
                j += 1
            if a is not None:
                qa.append((q, a))
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return qa


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def compute_entity_f1(q_entities: List[str], a_entities: List[str]) -> float:
    q_set = {e.strip().lower() for e in q_entities if isinstance(e, str) and e.strip()}
    a_set = {e.strip().lower() for e in a_entities if isinstance(e, str) and e.strip()}

    if not q_set and not a_set:
        return 0.0

    tp = len(q_set & a_set)
    fp = len(a_set - q_set)
    fn = len(q_set - a_set)

    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


# ----------------------------
# Judge model and prompting
# ----------------------------

LOCAL_SYSTEM_PROMPT = (
    "You are an assistant that analyzes the FORM / STYLE of a single QA turn.\n"
    "Your job is NOT to judge factual correctness, but ONLY to see whether the answer's\n"
    "style and format match what the question is asking for.\n"
    "\n"
    "Given a user question and an assistant answer, you MUST output a JSON object with:\n"
    "  - \"q_entities\": list of key entities in the user question.\n"
    "  - \"a_entities\": list of key entities in the assistant answer.\n"
    "  - \"style_match_score\": integer in {0, 1, 2}:\n"
    "       * 2 = The answer's style/format clearly matches the request type and constraints.\n"
    "       * 1 = Partially matches with minor format violations.\n"
    "       * 0 = Clearly mismatched style or ignores explicit constraints.\n"
    "  - \"style_comment\": a short English explanation (1-2 sentences).\n"
    "\n"
    "Important:\n"
    "- Focus ONLY on style / format compatibility with the question.\n"
    "- Do NOT judge factual correctness or safety.\n"
    "- Output ONLY one JSON object, no extra text.\n"
)

def build_turn_prompt(question: str, answer: str) -> str:
    return (
        "Analyze the following QA turn. Focus ONLY on style/format matching.\n\n"
        f"User question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n"
    )


@dataclass
class GenConfig:
    max_input_tokens: int = 2000
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


def load_judge(model_name: str, dtype: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    return tok, model


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def analyze_turn_batch(
    tokenizer,
    model,
    qa_batch: List[Tuple[str, str]],
    gen: GenConfig,
) -> List[Optional[Dict[str, Any]]]:
    if not qa_batch:
        return []

    chats: List[str] = []
    for q, a in qa_batch:
        messages = [
            {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_turn_prompt(q or "", a or "")},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chats.append(chat_text)

    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=gen.max_input_tokens,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    do_sample = gen.temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
        )

    results: List[Optional[Dict[str, Any]]] = []
    for i in range(out.size(0)):
        gen_ids = out[i, prompt_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        obj = parse_json_from_text(raw)
        if obj is None:
            results.append(None)
            continue

        style = safe_int(obj.get("style_match_score", 0), default=0)
        style = max(0, min(2, style))
        q_entities = obj.get("q_entities", [])
        a_entities = obj.get("a_entities", [])
        comment = obj.get("style_comment", "")

        results.append(
            {
                "q_entities": q_entities if isinstance(q_entities, list) else [],
                "a_entities": a_entities if isinstance(a_entities, list) else [],
                "style_match_score": style,
                "style_comment": str(comment),
                "raw": raw,
            }
        )

    return results


# ----------------------------
# Local scoring and selection
# ----------------------------

@dataclass
class LocalAgg:
    n_turns: int
    avg_style_score: float          # 0..2
    avg_entity_f1: float            # 0..1
    low_style_ratio: float
    high_style_ratio: float
    s_form: float                   # 0..2
    s_entity: float                 # 0..1
    local_score: float              # 0..1


def compute_local_for_dialogues(
    items: List[Dict[str, Any]],
    tokenizer,
    model,
    gen: GenConfig,
    batch_size: int,
    max_turns_per_dialog: Optional[int],
) -> Tuple[Dict[Any, LocalAgg], Dict[Any, List[Dict[str, Any]]]]:
    """
    Returns:
      scores_by_conv: conv_id -> LocalAgg
      turns_by_conv:  conv_id -> list of per-turn dicts
    """
    flat_turns: List[Tuple[Any, int, str, str]] = []
    for it in items:
        cid = it["conv_id"]
        qa = extract_qa_pairs(it["dialog"])
        if not qa:
            continue
        if max_turns_per_dialog is not None:
            qa = qa[:max_turns_per_dialog]
        for t_idx, (q, a) in enumerate(qa):
            flat_turns.append((cid, t_idx, q, a))

    turns_by_conv: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    for start in tqdm(range(0, len(flat_turns), batch_size), desc="Local-scoring"):
        chunk = flat_turns[start : start + batch_size]
        qa_batch = [(q, a) for (_, _, q, a) in chunk]
        res = analyze_turn_batch(tokenizer, model, qa_batch, gen)

        for (cid, t_idx, _, _), r in zip(chunk, res):
            if r is None:
                continue
            f1 = compute_entity_f1(r["q_entities"], r["a_entities"])
            r["entity_f1"] = f1
            r["turn_idx"] = t_idx
            turns_by_conv[cid].append(r)

    scores_by_conv: Dict[Any, LocalAgg] = {}
    for cid, tlist in turns_by_conv.items():
        if not tlist:
            continue
        style_scores = [x["style_match_score"] for x in tlist]
        ent_f1s = [x["entity_f1"] for x in tlist]

        n_turns = len(tlist)
        avg_style = float(np.mean(style_scores))
        avg_ent = float(np.mean(ent_f1s))
        low_ratio = float(sum(s == 0 for s in style_scores) / n_turns)
        high_ratio = float(sum(s == 2 for s in style_scores) / n_turns)

        style_norm = avg_style / 2.0
        local_score = 0.5 * style_norm + 0.5 * avg_ent

        scores_by_conv[cid] = LocalAgg(
            n_turns=n_turns,
            avg_style_score=avg_style,
            avg_entity_f1=avg_ent,
            low_style_ratio=low_ratio,
            high_style_ratio=high_ratio,
            s_form=avg_style,
            s_entity=avg_ent,
            local_score=float(local_score),
        )

    return scores_by_conv, dict(turns_by_conv)


def proportional_bin_budget(total_budget: int, bin_sizes: Dict[Any, int]) -> Dict[Any, int]:
    total = sum(bin_sizes.values())
    if total == 0:
        return {k: 0 for k in bin_sizes}
    raw = {k: total_budget * (v / total) for k, v in bin_sizes.items()}
    mk = {k: int(np.floor(x)) for k, x in raw.items()}
    # Distribute remaining budget by largest fractional parts
    remain = total_budget - sum(mk.values())
    if remain > 0:
        frac = sorted(((k, raw[k] - mk[k]) for k in mk), key=lambda x: x[1], reverse=True)
        for i in range(remain):
            mk[frac[i % len(frac)][0]] += 1
    return mk


def local_select(
    items: List[Dict[str, Any]],
    scores_by_conv: Dict[Any, LocalAgg],
    tau_form: float,
    total_budget: int,
) -> List[Dict[str, Any]]:
    """
    Selection rule:
      1) Filter by s_form >= tau_form
      2) Within each bin, rank by s_entity desc, then local_score desc
      3) Allocate per-bin budget proportionally, and take top mk
    """
    by_bin: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        cid = it["conv_id"]
        bin_id = it.get("bin_id", None)
        if bin_id is None:
            raise ValueError("Missing bin_id/cluster_id in candidates. Provide it in the input records.")
        agg = scores_by_conv.get(cid)
        if agg is None:
            continue
        if agg.s_form < tau_form:
            continue
        by_bin[bin_id].append(it)

    bin_sizes = {b: len(v) for b, v in by_bin.items()}
    mk = proportional_bin_budget(total_budget, bin_sizes)

    selected: List[Dict[str, Any]] = []
    for b, lst in by_bin.items():
        lst_sorted = sorted(
            lst,
            key=lambda x: (
                scores_by_conv[x["conv_id"]].s_entity,
                scores_by_conv[x["conv_id"]].local_score,
            ),
            reverse=True,
        )
        take = mk.get(b, 0)
        if take <= 0:
            continue
        selected.extend(lst_sorted[:take])

    # If rounding yields slightly more, truncate deterministically by local_score
    if len(selected) > total_budget:
        selected = sorted(
            selected, key=lambda x: scores_by_conv[x["conv_id"]].local_score, reverse=True
        )[:total_budget]

    return selected


# ----------------------------
# Distributed helpers
# ----------------------------

def init_distributed() -> Tuple[int, int]:
    """
    Returns (world_size, rank). Safe for single-process runs.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and dist is not None and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return world_size, rank


def barrier():
    if dist is not None and dist.is_initialized():
        dist.barrier()


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MDS Local Stage")
    p.add_argument("--input_jsonl", type=str, required=True, help="Candidate dialogues JSONL (from global stage).")
    p.add_argument("--output_jsonl", type=str, required=True, help="Selected dialogues JSONL.")
    p.add_argument("--output_ids_json", type=str, required=True, help="Selected conv_id list JSON.")
    p.add_argument("--output_scores_jsonl", type=str, default="", help="Optional: save per-dialog scores JSONL.")

    p.add_argument("--judge_model", type=str, required=True, help="HF path or repo id of the judge model.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--device", type=str, default="cuda", help="Device, e.g., cuda, cuda:0, cpu.")

    p.add_argument("--local_batch_size", type=int, default=64, help="Turn-level batch size for judge inference.")
    p.add_argument("--max_turns_per_dialog", type=int, default=-1, help="If >0, only score first N QA turns.")
    p.add_argument("--max_input_tokens", type=int, default=2000)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--tau_form", type=float, default=1.0, help="Form threshold on avg_style_score (0..2).")
    p.add_argument("--budget", type=int, default=10000, help="Total final selection budget.")
    p.add_argument("--keep_turn_details", action="store_true", help="If set, include turn details in output_jsonl.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    world_size, rank = init_distributed()
    device = args.device
    if device == "cuda" and torch.cuda.is_available() and world_size > 1:
        # Map rank to a local cuda device index when torchrun is used.
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = f"cuda:{local_rank}"

    max_turns = None if args.max_turns_per_dialog is None or args.max_turns_per_dialog <= 0 else args.max_turns_per_dialog

    gen = GenConfig(
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Load candidates
    all_items: List[Dict[str, Any]] = []
    for ex in read_jsonl(args.input_jsonl):
        it = normalize_dialog(ex)
        all_items.append(it)

    # Shard dialogues across ranks
    items = all_items[rank::world_size]

    # Load judge
    tokenizer, model = load_judge(args.judge_model, args.dtype, device)

    # Local scoring
    scores_by_conv, turns_by_conv = compute_local_for_dialogues(
        items=items,
        tokenizer=tokenizer,
        model=model,
        gen=gen,
        batch_size=args.local_batch_size,
        max_turns_per_dialog=max_turns,
    )

    # Save rank shard scores (optional)
    shard_dir = os.path.dirname(args.output_jsonl) or "."
    shard_scores_path = ""
    if args.output_scores_jsonl:
        shard_scores_path = os.path.join(
            shard_dir, f"_local_scores.rank{rank}.jsonl"
        )
        def score_records():
            for cid, agg in scores_by_conv.items():
                rec = {
                    "conv_id": cid,
                    "bin_id": None,
                    "scores": {
                        "n_turns": agg.n_turns,
                        "avg_style_score": agg.avg_style_score,
                        "avg_entity_f1": agg.avg_entity_f1,
                        "low_style_ratio": agg.low_style_ratio,
                        "high_style_ratio": agg.high_style_ratio,
                        "s_form": agg.s_form,
                        "s_entity": agg.s_entity,
                        "local_score": agg.local_score,
                    },
                }
                yield rec
        write_jsonl(shard_scores_path, score_records())

    # Merge scores on rank 0
    barrier()
    if rank != 0:
        return

    merged_scores_by_conv: Dict[Any, LocalAgg] = {}
    merged_turns_by_conv: Dict[Any, List[Dict[str, Any]]] = {}

    if world_size == 1:
        merged_scores_by_conv = scores_by_conv
        merged_turns_by_conv = turns_by_conv
    else:
        # Reload all shards by re-running normalize to keep ids consistent
        # Each rank processed disjoint dialogue subsets, so score keys do not overlap.
        for r in range(world_size):
            # Load scores shard if available, otherwise fall back to in-memory for rank 0 only.
            if r == 0:
                merged_scores_by_conv.update(scores_by_conv)
                merged_turns_by_conv.update(turns_by_conv)
                continue

            # Rank r does not keep in-memory data on rank 0, so read its shard.
            # For safety, require output_scores_jsonl when using multi-process.
            if not args.output_scores_jsonl:
                raise RuntimeError("For multi-process runs, set --output_scores_jsonl to enable shard merging.")

            pth = os.path.join(shard_dir, f"_local_scores.rank{r}.jsonl")
            for rec in read_jsonl(pth):
                cid = rec["conv_id"]
                sc = rec["scores"]
                merged_scores_by_conv[cid] = LocalAgg(
                    n_turns=int(sc["n_turns"]),
                    avg_style_score=float(sc["avg_style_score"]),
                    avg_entity_f1=float(sc["avg_entity_f1"]),
                    low_style_ratio=float(sc["low_style_ratio"]),
                    high_style_ratio=float(sc["high_style_ratio"]),
                    s_form=float(sc["s_form"]),
                    s_entity=float(sc["s_entity"]),
                    local_score=float(sc["local_score"]),
                )

        # Turn details are not merged across ranks by default to avoid large IO.
        # If needed, run single-process or extend this script.

    # Selection uses the full candidate list (rank 0 has it)
    # Only dialogues with available scores participate.
    selected_norm = local_select(
        items=all_items,
        scores_by_conv=merged_scores_by_conv,
        tau_form=args.tau_form,
        total_budget=args.budget,
    )

    selected_ids = [it["conv_id"] for it in selected_norm]

    # Build output records
    output_records: List[Dict[str, Any]] = []
    for it in selected_norm:
        raw = it["_raw"]
        cid = it["conv_id"]
        bin_id = it.get("bin_id", None)
        agg = merged_scores_by_conv.get(cid)
        out = dict(raw)
        out["conv_id"] = cid
        out["bin_id"] = bin_id
        if agg is not None:
            out["local_scores"] = {
                "n_turns": agg.n_turns,
                "avg_style_score": agg.avg_style_score,
                "avg_entity_f1": agg.avg_entity_f1,
                "low_style_ratio": agg.low_style_ratio,
                "high_style_ratio": agg.high_style_ratio,
                "s_form": agg.s_form,
                "s_entity": agg.s_entity,
                "local_score": agg.local_score,
            }
        if args.keep_turn_details:
            out["local_turns"] = merged_turns_by_conv.get(cid, [])
        output_records.append(out)

    write_jsonl(args.output_jsonl, output_records)
    write_json(args.output_ids_json, selected_ids)

    if args.output_scores_jsonl:
        # Write a compact merged score file for selected ids
        merged_path = args.output_scores_jsonl
        def merged_records():
            for cid in selected_ids:
                agg = merged_scores_by_conv.get(cid)
                if agg is None:
                    continue
                yield {
                    "conv_id": cid,
                    "scores": {
                        "n_turns": agg.n_turns,
                        "avg_style_score": agg.avg_style_score,
                        "avg_entity_f1": agg.avg_entity_f1,
                        "low_style_ratio": agg.low_style_ratio,
                        "high_style_ratio": agg.high_style_ratio,
                        "s_form": agg.s_form,
                        "s_entity": agg.s_entity,
                        "local_score": agg.local_score,
                    },
                }
        write_jsonl(merged_path, merged_records())

    # Cleanup shard files
    if args.output_scores_jsonl and world_size > 1:
        for r in range(world_size):
            pth = os.path.join(shard_dir, f"_local_scores.rank{r}.jsonl")
            if os.path.exists(pth):
                try:
                    os.remove(pth)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
