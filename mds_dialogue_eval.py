"""
Unified evaluation script for multi-turn dialogue prediction files.

Supported metrics (can be enabled or disabled):
1) LLM-EVAL (content / grammar / relevance / appropriateness / overall)
2) G-EVAL (coherence / naturalness / engagement / groundedness / overall)
3) Entity-F1 (GPT-based entity extraction with micro F1)
4) Embedding cosine similarity (sentence-transformers)
5) BLEU-3 (sacreBLEU)
6) ROUGE-L F1 (rouge_score)

The input file is assumed to be in JSONL format. Each line should contain at least:
{
  "idx": ...,
  "conv_id": ...,
  "turn_index": ...,
  "prompt": "<|start_header_id|>... a Llama 3 prompt string",
  "gold": "reference answer",
  "prediction": "model output"
}

After evaluation, each record is extended with fields such as:
- "llm_eval", "llm_raw" (if LLM-EVAL is enabled)
- "geval", "geval_raw" (if G-EVAL is enabled)
- "entities_gold", "entities_pred", "entity_tp", "entity_fp", "entity_fn"
  (if Entity-F1 is enabled)
and other metric-specific outputs.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional

from tqdm import tqdm
import numpy as np
import time
import openai
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============== 0. Basic configuration ==============

# OpenAI client. Update base_url and api_key for your setup.
client = openai.OpenAI(
    base_url = '',
    api_key = ''
)

# Models used for GPT-based evaluation
LLM_EVAL_MODEL = "gpt-4o-mini"
G_EVAL_MODEL = "gpt-4o-mini"
ENTITY_MODEL = "gpt-4o-mini"

# Number of worker threads. Adjust based on your account rate limits.
MAX_WORKERS_LLM = 40
MAX_WORKERS_ENTITY = 40
LLM_EVAL_TIMEOUT = 15   # Maximum wait time for one LLM-EVAL request.
G_EVAL_TIMEOUT = 15

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============== 1. General utility functions ==============

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_context_block_from_prompt(prompt_str: str) -> str:
    """
    Convert a Llama 3 prompt with <|start_header_id|> / <|eot_id|> markers
    into a readable multi-turn dialogue block.

    Output format:
    User: ...
    Assistant: ...
    System: ...
    """
    prompt_str = prompt_str or ""
    if "<|start_header_id|>" not in prompt_str:
        return prompt_str.strip()

    parts = prompt_str.split("<|start_header_id|>")
    lines = []
    for part in parts[1:]:
        # Example part format: 'user<|end_header_id|>\n\ncontent<|eot_id|>...'
        if "<|end_header_id|>" not in part:
            continue
        role_chunk, rest = part.split("<|end_header_id|>", 1)
        role = role_chunk.strip()
        if "<|eot_id|>" in rest:
            content, _ = rest.split("<|eot_id|>", 1)
        else:
            content = rest
        content = content.strip()
        if not content:
            continue
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = role.capitalize()
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


# ============== 2. LLM-EVAL ==============

# ============== 2. LLM-EVAL (direct 0-10 scoring) ==============

LLM_EVAL_SYSTEM_PROMPT = (
    "You are a STRICT dialogue evaluation assistant.\n"
    "You will evaluate a single model response given a dialogue context.\n"
    "\n"
    "You must rate the response on four dimensions:\n"
    "- content: correctness, informativeness, completeness, and conciseness of information.\n"
    "- grammar: fluency, grammar, clarity, naturalness, and conciseness of wording.\n"
    "- relevance: how well the response addresses the latest user message, stays on topic,\n"
    "  and avoids unnecessary or off-topic details.\n"
    "- appropriateness: safety, politeness, and instruction-following.\n"
    "\n"
    "VERY IMPORTANT:\n"
    "- Do NOT reward unnecessary verbosity.\n"
    "- If the response repeats itself, includes filler phrases (e.g., long preambles,\n"
    "  generic disclaimers, or obvious restatements), or adds text that does not\n"
    "  help answer the user's question, you MUST LOWER THE SCORES.\n"
    "- A shorter response that fully and clearly answers the question should receive\n"
    "  HIGHER scores than a much longer response that is equally correct but redundant.\n"
    "\n"
    "For EACH dimension, you MUST assign an INTEGER score from 0 to 10:\n"
    "- 0 = very bad: serious errors or largely unusable.\n"
    "- 2 = poor: many issues; only partially usable.\n"
    "- 4 = borderline: mixed quality with noticeable issues.\n"
    "- 6 = good: generally correct and appropriate but clearly improvable.\n"
    "- 8 = very good: high quality with only minor issues.\n"
    "- 10 = excellent: near human-expert quality; this should be RARE.\n"
    "\n"
    "You MUST use the FULL RANGE of scores when appropriate.\n"
    "Typical LLM responses that are acceptable but not outstanding should receive 2 or 6,\n"
    "NOT 8 or 10.\n"
    "\n"
    "First, you may briefly analyze the response for each dimension in free text.\n"
    "Be especially strict about unnecessary verbosity, repetition, and filler.\n"
    "\n"
    "Then, at the VERY END of your answer, output EXACTLY ONE line in the following format:\n"
    "FINAL_JSON: {\"content\": c, \"grammar\": g,\n"
    "             \"relevance\": r, \"appropriateness\": a}\n"
    "where c, g, r, a are INTEGERS in the range [0, 10].\n"
    "\n"
    "Do NOT output any other JSON objects besides this FINAL_JSON line.\n"
    "Do NOT use markdown code fences like ```.\n"
)


def build_llm_eval_user_prompt(context_block, model_response):
    return f"""
You are given a dialogue context and a model-generated response.

Carefully read the context and the response.

For each dimension (content, grammar, relevance, appropriateness),
assign an INTEGER score from 0 to 10, following the scale:

0 = very bad
2 = poor
4 = borderline
6 = good
8 = very good
10 = excellent (rare)

You MUST treat unnecessary verbosity, repetition, and filler as QUALITY ISSUES:
- Do not reward long preambles, generic disclaimers, or repeated information.
- Prefer responses that answer the question fully and clearly in a concise way.
- If the response could be made much shorter without losing important information,
  you should lower the scores.

At the END of your answer, output EXACTLY ONE line in the following format:

FINAL_JSON: {{"content": c, "grammar": g,
              "relevance": r, "appropriateness": a}}

where c, g, r, a are integers in [0, 10].

Context:
{context_block}

Model response:
{model_response}
""".strip()


def run_llm_eval(
    context_block,
    model_response,
    max_retries=2,
    timeout=LLM_EVAL_TIMEOUT,
):
    """
    Call the LLM-EVAL prompt for one sample and return
    (scores_dict, raw_text).

    Timeout handling is intentionally strict:
    - If a request times out, return None immediately for this sample.
    - Other errors can still be retried up to max_retries.
    """
    user_prompt = build_llm_eval_user_prompt(context_block, model_response)
    gpt_raw = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": LLM_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                timeout=timeout,
            )
            gpt_raw = resp.choices[0].message.content.strip()

            m = re.search(r'FINAL_JSON:\s*(\{.*\})', gpt_raw, re.DOTALL)
            if not m:
                # print(gpt_raw)
                # print('-'*20)
                # print(m)
                raise ValueError("No FINAL_JSON line found.")
            json_str = m.group(1).strip()
            scores_json = json.loads(json_str)

            def _get_score(d, key):
                v = d.get(key, None)
                if not isinstance(v, int):
                    raise ValueError(f"{key} must be int in [0, 10], got {v}")
                if v < 0 or v > 10:
                    raise ValueError(f"{key} out of range [0, 10], got {v}")
                return float(v)

            content = _get_score(scores_json, "content")
            grammar = _get_score(scores_json, "grammar")
            relevance = _get_score(scores_json, "relevance")
            appropriateness = _get_score(scores_json, "appropriateness")

            # Keep overall as the average of content, grammar, and relevance.
            overall = (content + grammar + relevance) / 3.0

            scores = {
                "content": content,
                "grammar": grammar,
                "relevance": relevance,
                "appropriateness": appropriateness,
                "overall": overall,
            }
            return scores, gpt_raw

        except Exception as e:
            msg = str(e).lower()
            print(f"[LLM-EVAL ERROR attempt {attempt}/{max_retries}] {e}")

            if "timed out" in msg or "timeout" in msg:
                print("[LLM-EVAL] timeout, skip this sample.")
                return None, gpt_raw or ""

            if attempt == max_retries:
                return None, gpt_raw or ""

            time.sleep(1.0)



# ============== 3. G-EVAL ==============

G_EVAL_SYSTEM_PROMPT = (
    "You are a STRICT dialogue-level evaluation assistant.\n"
    "You evaluate multi-turn conversations between a user and an assistant.\n"
    "\n"
    "You must rate the assistant's latest response on four dimensions:\n"
    "- coherence: how logically consistent and well-connected the response is "
    "  with the previous turns in the dialogue, without rambling.\n"
    "- naturalness: how fluent, human-like, stylistically appropriate, and concise the response is.\n"
    "- engagement: how interesting, proactive, and conversationally engaging the response is,\n"
    "  without resorting to unnecessary chit-chat or padding.\n"
    "- groundedness: how well the response is grounded in the given context, without hallucinating "
    "  unsupported facts or contradicting the dialogue.\n"
    "\n"
    "VERY IMPORTANT:\n"
    "- Do NOT reward unnecessary verbosity.\n"
    "- Long answers that repeat themselves, add generic filler, or provide off-topic\n"
    "  explanations should receive LOWER scores for coherence, naturalness, and engagement.\n"
    "- A shorter response that fits naturally into the dialogue and stays focused on the\n"
    "  user's needs should receive HIGHER scores than a much longer, padded response.\n"
    "\n"
    "For EACH dimension, you MUST assign an INTEGER score from 0 to 10:\n"
    "- 0 = very bad: serious issues; largely unusable.\n"
    "- 2 = poor: many issues; only partially usable.\n"
    "- 4 = borderline: mixed quality with noticeable issues.\n"
    "- 6 = good: generally fine but clearly improvable.\n"
    "- 8 = very good: high quality with only minor issues.\n"
    "- 10 = excellent: near human-expert quality; this should be RARE.\n"
    "\n"
    "You MUST use the FULL RANGE of scores when appropriate.\n"
    "Typical acceptable but not outstanding responses should receive 2 or 6,\n"
    "NOT 8 or 10.\n"
    "\n"
    "First, you may briefly analyze the response for each dimension in free text.\n"
    "Be especially strict about unnecessary verbosity, repetition, and filler.\n"
    "\n"
    "Then, at the VERY END of your answer, output EXACTLY ONE line in the following format:\n"
    "FINAL_JSON: {\"coherence\": c, \"naturalness\": n,\n"
    "             \"engagement\": e, \"groundedness\": g}\n"
    "where c, n, e, g are INTEGERS in the range [0, 10].\n"
    "\n"
    "Do NOT output any other JSON objects besides this FINAL_JSON line.\n"
    "Do NOT use markdown code fences like ```.\n"
)


def build_geval_user_prompt(context_block, model_response):
    return f"""
You are given a multi-turn dialogue context and a model-generated response
(from the assistant, corresponding to the last user message).

Carefully read the dialogue and the response.

For each dimension (coherence, naturalness, engagement, groundedness),
assign an INTEGER score from 0 to 10, following the scale:

0 = very bad
2 = poor
4 = borderline
6 = good
8 = very good
10 = excellent (rare)

You MUST treat unnecessary verbosity, repetition, and filler as QUALITY ISSUES:
- Do not reward long, rambling turns that could be much shorter.
- Penalize generic chit-chat or boilerplate text that does not move the conversation forward.
- Prefer responses that smoothly fit the dialogue, are focused and concise.

At the VERY END, output EXACTLY ONE line:

FINAL_JSON: {{"coherence": c, "naturalness": n,
              "engagement": e, "groundedness": g}}

where c, n, e, g are integers in [0, 10].

Context:
{context_block}

Model response:
{model_response}
""".strip()


def _llm_eval_worker(idx: int, context_block: str, pred: str):
    scores, raw = run_llm_eval(context_block, pred)
    return idx, scores, raw


def batch_run_llm_eval(
    context_blocks: List[str],
    preds: List[str],
    max_workers: int = MAX_WORKERS_LLM,
    desc: str = "LLM-EVAL",
):
    n = len(preds)
    results: List[Tuple[Optional[Dict[str, float]], str]] = [(None, "") for _ in range(n)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, (ctx, p) in enumerate(zip(context_blocks, preds)):
            futures.append(ex.submit(_llm_eval_worker, i, ctx, p))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, scores, raw = fut.result()
            results[idx] = (scores, raw)

    return results


def _geval_worker(idx: int, context_block: str, pred: str):
    scores, raw = run_geval(context_block, pred)
    return idx, scores, raw


def batch_run_geval(
    context_blocks: List[str],
    preds: List[str],
    max_workers: int = MAX_WORKERS_LLM,
    desc: str = "G-EVAL",
):
    n = len(preds)
    results: List[Tuple[Optional[Dict[str, float]], str]] = [(None, "") for _ in range(n)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, (ctx, p) in enumerate(zip(context_blocks, preds)):
            futures.append(ex.submit(_geval_worker, i, ctx, p))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, scores, raw = fut.result()
            results[idx] = (scores, raw)

    return results


def _entity_worker(idx: int, gold: str, pred: str):
    ents_gold, _ = extract_entities(gold)
    ents_pred, _ = extract_entities(pred)
    tp, fp, fn = compute_entity_counts(ents_gold, ents_pred)
    return idx, ents_gold, ents_pred, tp, fp, fn


def batch_extract_entities(
    golds: List[str],
    preds: List[str],
    max_workers: int = MAX_WORKERS_ENTITY,
    desc: str = "Entity-F1",
):
    n = len(golds)
    # ents_gold, ents_pred, tp, fp, fn
    results: List[Tuple[List[str], List[str], int, int, int]] = [
        ([], [], 0, 0, 0) for _ in range(n)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, (g, p) in enumerate(zip(golds, preds)):
            futures.append(ex.submit(_entity_worker, i, g, p))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, ents_g, ents_p, tp, fp, fn = fut.result()
            results[idx] = (ents_g, ents_p, tp, fp, fn)

    return results
    
def run_geval(context_block, model_response, max_retries=2, timeout=G_EVAL_TIMEOUT):
    """
    Score one sample with G-EVAL and return (scores_dict, raw_text).
    """
    user_prompt = build_geval_user_prompt(context_block, model_response)
    gpt_raw = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=G_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": G_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                timeout=timeout,
            )
            gpt_raw = resp.choices[0].message.content.strip()

            m = re.search(r'FINAL_JSON:\s*(\{.*\})', gpt_raw, re.DOTALL)
            if not m:
                # print(gpt_raw)
                # print('-'*20)
                # print(m)
                raise ValueError("No FINAL_JSON line found in G-EVAL.")
            json_str = m.group(1).strip()
            scores_json = json.loads(json_str)

            def _get_score(d, key):
                v = d.get(key, None)
                if not isinstance(v, int):
                    raise ValueError(f"{key} must be int in [0, 10], got {v}")
                if v < 0 or v > 10:
                    raise ValueError(f"{key} out of range [0, 10], got {v}")
                return float(v)

            coherence = _get_score(scores_json, "coherence")
            naturalness = _get_score(scores_json, "naturalness")
            engagement = _get_score(scores_json, "engagement")
            groundedness = _get_score(scores_json, "groundedness")

            overall = (coherence + naturalness + engagement + groundedness) / 4.0

            scores = {
                "coherence": coherence,
                "naturalness": naturalness,
                "engagement": engagement,
                "groundedness": groundedness,
                "overall": overall,
            }
            return scores, gpt_raw

        except Exception as e:
            msg = str(e).lower()
            print(f"[G-EVAL ERROR attempt {attempt}/{max_retries}] {e}")

            if "timed out" in msg or "timeout" in msg:
                print("[G-EVAL] timeout, skip this sample.")
                return None, gpt_raw or ""

            if attempt == max_retries:
                return None, gpt_raw or ""

            time.sleep(1.0)


# ============== 4. Entity-F1 (GPT-based entity extraction) ==============

ENTITY_SYSTEM_PROMPT = (
    "You are an information extraction assistant.\n"
    "Given a short answer text, you must extract key entities mentioned in the text.\n"
    "Entities can be people, organizations, locations, dates, times, events, "
    "diseases, products, important concepts, etc.\n"
    "Output ONLY a JSON object of the form:\n"
    "{\"entities\": [\"entity1\", \"entity2\", ...]}\n"
    "Use short canonical phrases where possible. "
    "Do NOT include explanations or any other text."
)


def build_entity_user_prompt(text: str) -> str:
    return f"Extract key entities from the following answer text.\n\nText:\n{text}"


def parse_entity_json(raw: str) -> List[str]:
    """
    Parse a JSON object like {"entities": [...]} from the model output.
    """
    raw = raw.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        ents = obj.get("entities", [])
        if not isinstance(ents, list):
            return []
        cleaned = []
        for e in ents:
            if not isinstance(e, str):
                continue
            e = e.strip()
            if e:
                cleaned.append(e)
        return cleaned
    except Exception:
        return []


def extract_entities(text: str, max_retries: int = 2) -> Tuple[List[str], str]:
    """
    Extract entities from one text with GPT and return
    (entity_list, raw_output).
    """
    text = text or ""
    if not text.strip():
        return [], ""

    user_prompt = build_entity_user_prompt(text)
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=ENTITY_MODEL,
                messages=[
                    {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            ents = parse_entity_json(raw)
            return ents, raw
        except Exception as e:
            print(f"[ENTITY ERROR attempt {attempt}/{max_retries}] {e}")
            if attempt == max_retries:
                return [], raw or ""


def compute_entity_counts(gold_ents: List[str], pred_ents: List[str]) -> Tuple[int, int, int]:
    """
    Given gold and predicted entity lists, return (TP, FP, FN)
    for micro-F1 computation.

    Matching is case-insensitive and strips surrounding whitespace.
    """
    gs = {e.strip().lower() for e in gold_ents if e.strip()}
    ps = {e.strip().lower() for e in pred_ents if e.strip()}
    tp = len(gs & ps)
    fp = len(ps - gs)
    fn = len(gs - ps)
    return tp, fp, fn


# ============== 5. Embedding cosine / BLEU-3 / ROUGE-L ==============

_embed_model = None
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def compute_semantic_similarity_list(
    preds: List[str],
    golds: List[str],
    batch_size: int = 64
) -> List[float]:
    """
    Compute sentence-transformer cosine similarity for each sample.
    """
    model = get_embed_model()
    assert len(preds) == len(golds)
    sims: List[float] = []
    for i in range(0, len(preds), batch_size):
        pred_batch = preds[i:i+batch_size]
        gold_batch = golds[i:i+batch_size]
        emb_pred = model.encode(pred_batch, convert_to_tensor=True, show_progress_bar=False)
        emb_gold = model.encode(gold_batch, convert_to_tensor=True, show_progress_bar=False)
        cos = F.cosine_similarity(emb_pred, emb_gold)
        sims.extend([float(x) for x in cos.cpu().tolist()])
    return sims


def compute_avg_semantic_similarity(
    preds: List[str],
    golds: List[str],
    batch_size: int = 64
) -> float:
    """
    Average cosine similarity. Kept for backward compatibility.
    """
    sims = compute_semantic_similarity_list(preds, golds, batch_size=batch_size)
    if not sims:
        return 0.0
    return float(np.mean(sims))


_bleu_metric = BLEU(effective_order=True)

def compute_bleu3_list(preds: List[str], golds: List[str]) -> List[float]:
    """
    Return the per-sample BLEU-3 list.
    """
    assert len(preds) == len(golds)
    scores: List[float] = []
    for p, g in zip(preds, golds):
        bleu_result = _bleu_metric.sentence_score(p, [g])
        scores.append(float(bleu_result.precisions[2] / 100.0))
    return scores


def compute_bleu3(preds: List[str], golds: List[str]) -> float:
    """
    Average BLEU-3. Kept for backward compatibility.
    """
    scores = compute_bleu3_list(preds, golds)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_rougeL_list(preds: List[str], golds: List[str]) -> List[float]:
    """
    Return the per-sample ROUGE-L F1 list.
    """
    assert len(preds) == len(golds)
    scores = []
    for p, g in zip(preds, golds):
        score = _rouge_scorer.score(g, p)
        scores.append(float(score["rougeL"].fmeasure))
    return scores


def compute_rougeL(preds: List[str], golds: List[str]) -> float:
    """
    Average ROUGE-L F1. Kept for backward compatibility.
    """
    scores = compute_rougeL_list(preds, golds)
    if not scores:
        return 0.0
    return float(np.mean(scores))


# ============== 6. Main evaluation pipeline for multiple files ==============

def eval_file(path: str) -> Dict[str, Any]:
    """
    Run the selected metrics on one JSONL file.

    LLM-EVAL, G-EVAL, and Entity-F1 are executed with multithreading.
    """
    file_name = os.path.basename(path)
    print(f"\n=== Evaluating file: {file_name} ===")

    records = load_jsonl(path)
    print(f"Loaded {len(records)} records.")
    if MAX_RECORDS_PER_FILE is not None and len(records) > MAX_RECORDS_PER_FILE:
        records = records[:MAX_RECORDS_PER_FILE]
        print(f"Truncated to first {len(records)} records for evaluation.")

    n = len(records)
    if n == 0:
        print("[WARN] No records, skip.")
        return {
            "file": file_name,
            "file_path": path,
            "n_samples": 0,
        }

    # Pre-extract context, gold answers, and predictions
    context_blocks: List[str] = []
    golds: List[str] = []
    preds: List[str] = []
    for rec in records:
        prompt_str = rec.get("prompt", "") or ""
        context_blocks.append(build_context_block_from_prompt(prompt_str))
        golds.append(rec.get("gold", "") or "")
        preds.append(rec.get("prediction", "") or "")

    # ---------- 1) LLM-EVAL (multithreaded) ----------
    if USE_LLM_EVAL:
        llm_batch = batch_run_llm_eval(
            context_blocks, preds,
            max_workers=MAX_WORKERS_LLM,
            desc=f"{file_name} | LLM-EVAL",
        )
    else:
        llm_batch = [(None, "") for _ in range(n)]

    # ---------- 2) G-EVAL (multithreaded) ----------
    if USE_G_EVAL:
        geval_batch = batch_run_geval(
            context_blocks, preds,
            max_workers=MAX_WORKERS_LLM,
            desc=f"{file_name} | G-EVAL",
        )
    else:
        geval_batch = [(None, "") for _ in range(n)]

    # ---------- 3) Entity-F1 (multithreaded) ----------
    if USE_ENTITY_F1:
        entity_batch = batch_extract_entities(
            golds, preds,
            max_workers=MAX_WORKERS_ENTITY,
            desc=f"{file_name} | Entity-F1",
        )
    else:
        entity_batch = [([], [], 0, 0, 0) for _ in range(n)]

    # ---------- 4) Text similarity metrics (CPU-side batched processing) ----------
    if USE_EMBED_COS:
        embed_cos_list = compute_semantic_similarity_list(preds, golds)
    else:
        embed_cos_list = []

    if USE_BLEU3:
        bleu3_list = compute_bleu3_list(preds, golds)
    else:
        bleu3_list = []

    if USE_ROUGEL:
        rougeL_list = compute_rougeL_list(preds, golds)
    else:
        rougeL_list = []

    # ---------- 5) Aggregate per-sample outputs and summary statistics ----------
    out_records: List[Dict[str, Any]] = []

    # LLM-EVAL statistics
    sum_llm_content = sum_llm_grammar = 0.0
    sum_llm_relevance = sum_llm_appropriateness = 0.0
    sum_llm_overall = 0.0
    n_llm = 0

    # G-EVAL statistics
    sum_g_coh = sum_g_nat = sum_g_eng = sum_g_grd = 0.0
    sum_g_overall = 0.0
    n_geval = 0

    # Entity-F1 statistics
    global_TP = global_FP = global_FN = 0

    for i, rec in enumerate(records):
        out_rec = dict(rec)
        gold = golds[i]
        pred = preds[i]

        # ---------- LLM-EVAL ----------
        llm_scores, llm_raw = llm_batch[i]
        if USE_LLM_EVAL:
            out_rec["llm_eval"] = llm_scores
            out_rec["llm_raw"] = llm_raw
            if llm_scores is not None:
                sum_llm_content += llm_scores["content"]
                sum_llm_grammar += llm_scores["grammar"]
                sum_llm_relevance += llm_scores["relevance"]
                sum_llm_appropriateness += llm_scores["appropriateness"]
                sum_llm_overall += llm_scores["overall"]
                n_llm += 1
        else:
            out_rec["llm_eval"] = None
            out_rec["llm_raw"] = ""

        # ---------- G-EVAL ----------
        geval_scores, geval_raw = geval_batch[i]
        if USE_G_EVAL:
            out_rec["geval"] = geval_scores
            out_rec["geval_raw"] = geval_raw
            if geval_scores is not None:
                sum_g_coh += geval_scores["coherence"]
                sum_g_nat += geval_scores["naturalness"]
                sum_g_eng += geval_scores["engagement"]
                sum_g_grd += geval_scores["groundedness"]
                sum_g_overall += geval_scores["overall"]
                n_geval += 1
        else:
            out_rec["geval"] = None
            out_rec["geval_raw"] = ""

        # ---------- Entity-F1 ----------
        ents_gold, ents_pred, tp, fp, fn = entity_batch[i]
        if USE_ENTITY_F1:
            global_TP += tp
            global_FP += fp
            global_FN += fn

            out_rec["entities_gold"] = ents_gold
            out_rec["entities_pred"] = ents_pred
            out_rec["entity_tp"] = tp
            out_rec["entity_fp"] = fp
            out_rec["entity_fn"] = fn
        else:
            out_rec["entities_gold"] = None
            out_rec["entities_pred"] = None
            out_rec["entity_tp"] = 0
            out_rec["entity_fp"] = 0
            out_rec["entity_fn"] = 0

        # ---------- Text similarity ----------
        if USE_EMBED_COS and embed_cos_list:
            out_rec["embed_cos"] = float(embed_cos_list[i])
        else:
            out_rec["embed_cos"] = None

        if USE_BLEU3 and bleu3_list:
            out_rec["bleu3"] = float(bleu3_list[i])
        else:
            out_rec["bleu3"] = None

        if USE_ROUGEL and rougeL_list:
            out_rec["rougeL"] = float(rougeL_list[i])
        else:
            out_rec["rougeL"] = None

        out_records.append(out_rec)

    # ---------- 6) Write per-sample outputs ----------
    base, ext = os.path.splitext(path)
    out_path = f"{base}_all_eval_{EVAL_VERSION}.jsonl"
    save_jsonl(out_path, out_records)
    print(f"[file] per-sample eval written to: {out_path}")

    summary: Dict[str, Any] = {
        "file": file_name,
        "file_path": path,
        "n_samples": len(out_records),
    }

    # LLM-EVAL averages
    if USE_LLM_EVAL:
        if n_llm > 0:
            avg_llm_content = sum_llm_content / n_llm
            avg_llm_grammar = sum_llm_grammar / n_llm
            avg_llm_relevance = sum_llm_relevance / n_llm
            avg_llm_appropriateness = sum_llm_appropriateness / n_llm
            avg_llm_overall = sum_llm_overall / n_llm
        else:
            avg_llm_content = avg_llm_grammar = 0.0
            avg_llm_relevance = avg_llm_appropriateness = 0.0
            avg_llm_overall = 0.0

        summary.update({
            "llm_content": avg_llm_content,
            "llm_grammar": avg_llm_grammar,
            "llm_relevance": avg_llm_relevance,
            "llm_appropriateness": avg_llm_appropriateness,
            "llm_overall": avg_llm_overall,
        })

    # G-EVAL averages
    if USE_G_EVAL:
        if n_geval > 0:
            avg_g_coh = sum_g_coh / n_geval
            avg_g_nat = sum_g_nat / n_geval
            avg_g_eng = sum_g_eng / n_geval
            avg_g_grd = sum_g_grd / n_geval
            avg_g_overall = sum_g_overall / n_geval
        else:
            avg_g_coh = avg_g_nat = avg_g_eng = avg_g_grd = 0.0
            avg_g_overall = 0.0

        summary.update({
            "geval_coherence": avg_g_coh,
            "geval_naturalness": avg_g_nat,
            "geval_engagement": avg_g_eng,
            "geval_groundedness": avg_g_grd,
            "geval_overall": avg_g_overall,
        })

    # Entity-F1 (micro)
    if USE_ENTITY_F1:
        if global_TP + global_FP + global_FN == 0:
            entity_precision = entity_recall = entity_f1 = 0.0
        else:
            entity_precision = global_TP / (global_TP + global_FP + 1e-8)
            entity_recall = global_TP / (global_TP + global_FN + 1e-8)
            entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall + 1e-8)

        summary.update({
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
        })

    # Overall text similarity metrics
    if USE_EMBED_COS:
        if embed_cos_list:
            summary["embed_cos"] = float(np.mean(embed_cos_list))
        else:
            summary["embed_cos"] = 0.0
    if USE_BLEU3:
        if bleu3_list:
            summary["bleu3"] = float(np.mean(bleu3_list))
        else:
            summary["bleu3"] = 0.0
    if USE_ROUGEL:
        if rougeL_list:
            summary["rougeL"] = float(np.mean(rougeL_list))
        else:
            summary["rougeL"] = 0.0

    # Print a compact summary of the enabled metrics
    msg = f"[STATS] {file_name} | n={len(out_records)}"
    if USE_LLM_EVAL:
        msg += f" | LLM-overall={summary['llm_overall']:.3f}"
    if USE_G_EVAL:
        msg += f" | G-overall={summary['geval_overall']:.3f}"
    if USE_ENTITY_F1:
        msg += f" | Entity-F1={summary['entity_f1']:.3f}"
    if USE_EMBED_COS:
        msg += f" | Cos={summary['embed_cos']:.4f}"
    if USE_BLEU3:
        msg += f" | BLEU-3={summary['bleu3']:.4f}"
    if USE_ROUGEL:
        msg += f" | ROUGE-L={summary['rougeL']:.4f}"
    print(msg)

    return summary

# Maximum number of records to evaluate per file
MAX_RECORDS_PER_FILE = 300

# Version tag used in output filenames
EVAL_VERSION = "_direct_socre_v3"

# ============== Metric switches ==============
# Enable or disable metrics as needed, especially the GPT-based ones.

USE_LLM_EVAL = True        # GPT-based: content / grammar / relevance / appropriateness
USE_G_EVAL = True          # GPT-based: coherence / naturalness / engagement / groundedness
USE_ENTITY_F1 = True       # GPT entity extraction + F1
USE_EMBED_COS = True       # sentence-transformers cosine similarity
USE_ROUGEL = True          # ROUGE-L F1
USE_BLEU3 = False          # sacreBLEU-3

# Directory containing prediction files.
INPUT_DIR = "/your_path/eval_preds"

# Files to be evaluated. These are filenames without directory prefixes.
FILE_LIST = [
    "pred_TopDial.jsonl",
    "pred_ConsistentChat.jsonl",
    "pred_MT_eval.jsonl",
]



def main() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for fname in FILE_LIST:
        path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] file not found: {path}")
            continue
        summary = eval_file(path)
        summaries.append(summary)

    if summaries:
        summary_path = os.path.join(INPUT_DIR, f"eval_summary{EVAL_VERSION}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[file] summary written to: {summary_path}")


if __name__ == "__main__":
    main()
