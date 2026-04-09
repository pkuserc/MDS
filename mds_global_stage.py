#!/usr/bin/env python3
"""
MDS Global Stage (candidate pool construction).
Usage:

python mds_global_stage.py \
  --input_jsonl data/dialog_pool.jsonl \
  --output_jsonl outputs/mds_global_selected.jsonl \
  --output_ids_json outputs/mds_global_selected_ids.json \
  --n_clusters 1000 \
  --retain_ratio 0.5 \
  --min_keep_per_cluster 5 \
  --lambda_mmr 0.5

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


# -----------------------------
# IO helpers
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {ln}: {e}") from e
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Dialogue normalization
# -----------------------------

def get_dialog_id(item: Dict[str, Any], fallback: int) -> Any:
    if "conv_id" in item:
        return item["conv_id"]
    if "id" in item:
        return item["id"]
    return fallback


def get_turns(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("messages", "dialog", "conversation"):
        v = item.get(k)
        if isinstance(v, list):
            return v
    return []


def get_turn_role(turn: Dict[str, Any]) -> str:
    r = turn.get("role", "")
    if not isinstance(r, str):
        r = ""
    return r.strip().lower()


def get_turn_text(turn: Dict[str, Any]) -> str:
    # Common fields: content (ShareGPT-like), text (some corpora)
    t = turn.get("content")
    if t is None:
        t = turn.get("text")
    if not isinstance(t, str):
        return ""
    return t.strip()


def estimate_len(text: str) -> int:
    # Lightweight length proxy for weighting.
    if not text:
        return 0
    return max(1, len(text.split()))


# -----------------------------
# Embedding
# -----------------------------

def get_embed_model(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    if device is None:
        return SentenceTransformer(model_name)
    return SentenceTransformer(model_name, device=device)


def encode_single_dialog(
    dialog_item: Dict[str, Any],
    dialog_id: Any,
    model: SentenceTransformer,
) -> Optional[np.ndarray]:
    turns = get_turns(dialog_item)

    user_texts: List[str] = []
    user_weights: List[int] = []

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = get_turn_role(turn)
        if role != "user":
            continue
        text = get_turn_text(turn)
        if not text:
            continue
        user_texts.append(text)
        user_weights.append(estimate_len(text))

    if not user_texts:
        return None

    weights = np.asarray(user_weights, dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    emb_users = model.encode(
        user_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )  # (T, D)

    v = (emb_users * weights[:, None]).sum(axis=0)  # (D,)
    return v.astype(np.float32)


def compute_dialog_embeddings(
    dialogs: List[Dict[str, Any]],
    embed_model_name: str,
    device: Optional[str] = None,
) -> Tuple[List[Any], np.ndarray, Dict[Any, Dict[str, Any]]]:
    model = get_embed_model(embed_model_name, device=device)

    conv_ids: List[Any] = []
    emb_list: List[np.ndarray] = []
    id2dialog: Dict[Any, Dict[str, Any]] = {}

    for i, item in enumerate(tqdm(dialogs, desc="Embedding dialogues")):
        cid = get_dialog_id(item, fallback=i)
        emb = encode_single_dialog(item, cid, model)
        if emb is None:
            continue
        conv_ids.append(cid)
        emb_list.append(emb)
        id2dialog[cid] = item

    if not emb_list:
        dim = int(model.get_sentence_embedding_dimension())
        return [], np.zeros((0, dim), dtype=np.float32), {}

    emb_mat = np.stack(emb_list, axis=0).astype(np.float32)  # (N, D)
    return conv_ids, emb_mat, id2dialog


# -----------------------------
# Clustering (semantic bins)
# -----------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def cluster_dialogs(
    conv_ids: List[Any],
    emb_mat: np.ndarray,
    n_clusters: Optional[int],
    batch_size: int,
    max_iter: int,
    random_state: int,
) -> Dict[str, Any]:
    emb = np.asarray(emb_mat, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("emb_mat must be 2D (N, D).")
    N, D = emb.shape

    emb_norm = l2_normalize(emb)

    if n_clusters is None:
        n_clusters = max(8, int(np.sqrt(N / 2.0)))

    kmeans = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        random_state=int(random_state),
        n_init="auto",
        verbose=0,
    )

    labels = kmeans.fit_predict(emb_norm)  # (N,)
    centers = kmeans.cluster_centers_      # (K, D)

    cluster_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, c in enumerate(labels):
        cluster_indices[int(c)].append(idx)

    cluster_conv_ids = {c: [conv_ids[i] for i in idxs] for c, idxs in cluster_indices.items()}
    cluster_sizes = {c: len(idxs) for c, idxs in cluster_indices.items()}

    return {
        "labels": labels,
        "centers": centers,
        "cluster_indices": dict(cluster_indices),
        "cluster_conv_ids": cluster_conv_ids,
        "cluster_sizes": cluster_sizes,
        "n_clusters": int(n_clusters),
        "N": int(N),
        "D": int(D),
    }


# -----------------------------
# Global selection
# -----------------------------

def global_mmr_select(
    conv_ids: List[Any],
    emb_mat: np.ndarray,
    cluster_result: Dict[str, Any],
    retain_ratio: float,
    min_keep_per_cluster: int,
    lambda_mmr: float,
    verbose: bool = False,
) -> Tuple[List[int], List[Any], List[int], Dict[int, List[int]]]:
    """
    Bin-wise MMR selection.

    For each cluster/bin:
    - discard bins with size <= min_keep_per_cluster
    - keep_k = max(min_keep_per_cluster, round(size * retain_ratio))
    - anchor = mean embedding of the bin (L2-normalized)
    - MMR score: lambda * sim(x, anchor) - (1-lambda) * max_{selected} sim(x, selected)
    """
    if not (0.0 < retain_ratio <= 1.0):
        raise ValueError("retain_ratio must be in (0, 1].")
    if not (0.0 <= lambda_mmr <= 1.0):
        raise ValueError("lambda_mmr must be in [0, 1].")

    emb = np.asarray(emb_mat, dtype=np.float32)
    N, _ = emb.shape
    if len(conv_ids) != N:
        raise ValueError("conv_ids length must match emb_mat rows.")

    emb_norm = l2_normalize(emb)

    cluster_indices: Dict[int, List[int]] = cluster_result["cluster_indices"]
    labels = cluster_result["labels"]

    selected_indices: List[int] = []
    selected_by_cluster: Dict[int, List[int]] = {}

    for cid, idx_list in cluster_indices.items():
        n = len(idx_list)
        if n == 0:
            continue

        # Discard small bins
        if n <= int(min_keep_per_cluster):
            if verbose:
                print(f"[global] cluster={cid} size={n} <= {min_keep_per_cluster}: dropped")
            continue

        keep_k = int(round(n * float(retain_ratio)))
        keep_k = max(int(min_keep_per_cluster), keep_k)
        keep_k = min(keep_k, n)

        v_bin = emb_norm[idx_list]  # (n, D)

        # Anchor (representativeness)
        anchor = v_bin.mean(axis=0)
        anchor = anchor / (np.linalg.norm(anchor) + 1e-8)
        anchor_sims = v_bin @ anchor  # (n,)

        first_local = int(np.argmax(anchor_sims))
        selected_local = [first_local]
        selected_vecs = [v_bin[first_local]]
        remaining = set(range(n))
        remaining.remove(first_local)

        while len(selected_local) < keep_k and remaining:
            selected_stack = np.stack(selected_vecs, axis=0)  # (m, D)
            best_score = -1e9
            best_li = None

            for li in remaining:
                sims_to_selected = selected_stack @ v_bin[li]  # (m,)
                max_sim = float(np.max(sims_to_selected)) if sims_to_selected.size else 0.0
                score = float(lambda_mmr) * float(anchor_sims[li]) - (1.0 - float(lambda_mmr)) * max_sim
                if score > best_score:
                    best_score = score
                    best_li = li

            if best_li is None:
                break
            selected_local.append(int(best_li))
            selected_vecs.append(v_bin[int(best_li)])
            remaining.remove(best_li)

        chosen_global = [idx_list[li] for li in selected_local]
        selected_by_cluster[int(cid)] = chosen_global
        selected_indices.extend(chosen_global)

        if verbose:
            print(f"[global] cluster={cid} size={n} kept={len(chosen_global)} ({len(chosen_global)/n:.2f})")

    selected_indices = sorted(set(int(i) for i in selected_indices))
    selected_conv_ids = [conv_ids[i] for i in selected_indices]
    selected_cluster_ids = [int(labels[i]) for i in selected_indices]

    return selected_indices, selected_conv_ids, selected_cluster_ids, selected_by_cluster


# -----------------------------
# Main
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MDS Global Stage: clustering + bin-wise MMR selection.")
    p.add_argument("--input_jsonl", type=str, required=True, help="Input dialogue pool in JSONL format.")
    p.add_argument("--output_jsonl", type=str, required=True, help="Output selected dialogues JSONL.")
    p.add_argument("--output_ids_json", type=str, required=True, help="Output selected dialogue IDs (JSON list).")

    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer name or path.")
    p.add_argument("--device", type=str, default=None, help="Embedding device (e.g., cpu, cuda).")

    p.add_argument("--n_clusters", type=int, default=1000, help="Number of semantic bins (k-means clusters).")
    p.add_argument("--cluster_batch_size", type=int, default=2048, help="MiniBatchKMeans batch size.")
    p.add_argument("--cluster_max_iter", type=int, default=100, help="MiniBatchKMeans max iterations.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for clustering.")

    p.add_argument("--retain_ratio", type=float, default=0.5, help="Per-bin retain ratio after global selection.")
    p.add_argument("--min_keep_per_cluster", type=int, default=5, help="Drop bins with size <= this value; also minimum keep for large bins.")
    p.add_argument("--lambda_mmr", type=float, default=0.5, help="MMR lambda balancing representativeness vs diversity.")
    p.add_argument("--verbose", action="store_true", help="Print per-cluster logs.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    dialogs = read_jsonl(args.input_jsonl)

    conv_ids, emb_mat, id2dialog = compute_dialog_embeddings(
        dialogs=dialogs,
        embed_model_name=args.embed_model,
        device=args.device,
    )
    if len(conv_ids) == 0:
        raise RuntimeError("No valid dialogues found (no user turns or empty pool).")

    cluster_result = cluster_dialogs(
        conv_ids=conv_ids,
        emb_mat=emb_mat,
        n_clusters=args.n_clusters,
        batch_size=args.cluster_batch_size,
        max_iter=args.cluster_max_iter,
        random_state=args.seed,
    )

    _, selected_conv_ids, _, _ = global_mmr_select(
        conv_ids=conv_ids,
        emb_mat=emb_mat,
        cluster_result=cluster_result,
        retain_ratio=args.retain_ratio,
        min_keep_per_cluster=args.min_keep_per_cluster,
        lambda_mmr=args.lambda_mmr,
        verbose=args.verbose,
    )

    selected_dialogs: List[Dict[str, Any]] = []
    missing = 0
    for cid in selected_conv_ids:
        d = id2dialog.get(cid)
        if d is None:
            missing += 1
            continue
        selected_dialogs.append(d)

    if missing > 0 and args.verbose:
        print(f"[global] warning: {missing} selected ids missing raw dialogue objects.")

    write_json(args.output_ids_json, selected_conv_ids)
    write_jsonl(args.output_jsonl, selected_dialogs)

    print(f"[global] input={len(dialogs)} embedded={len(conv_ids)} selected={len(selected_conv_ids)}")
    print(f"[global] saved ids -> {args.output_ids_json}")
    print(f"[global] saved dialogues -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
