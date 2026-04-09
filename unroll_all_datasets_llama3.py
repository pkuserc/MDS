import json
import time
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

start_time = time.time()

# =========================
# Configuration
# =========================

# Model path
MODEL_PATH = "/your_model_path/models/Meta-Llama-3-8B-Instruct"
# MODEL_PATH = "/your_model_path/models/Qwen3-8b"

# Input JSONL files
TRAIN_INPUT_PATHS = [
    "/outputs/MDS_seleted_baize.jsonl",
    "/outputs/MDS_seleted_banking.jsonl",
]

EVAL_INPUT_PATHS = [
    "/your_data_path/eval/ConsistentChat_use_270.jsonl",
    "/your_data_path/eval/dialogue_test_unseen_use.jsonl",
    "/your_data_path/eval/mt_eval_part_use.jsonl",
    "/your_data_path/eval/banking_test_300.jsonl",
]

# Output root directory
OUTPUT_ROOT = "/your_data_path/disk"

# Sequence length settings
MAX_BUILD_LEN = 3000
MAX_FILTER_LEN = 2000

# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    use_fast=False,
    trust_remote_code=True,
)

# Prefer <|eot_id|> as the end-of-message token.
if "<|eot_id|>" in tokenizer.all_special_tokens:
    EOT_TOKEN = "<|eot_id|>"
else:
    # Rare fallback in case tokenizer config was manually changed.
    EOT_TOKEN = tokenizer.eos_token


def build_input_with_context(context_messages, answer, max_len=8000):
    """
    Build one supervised sample for Llama 3 style chat SFT.

    Args:
        context_messages: list[{"role": "system"/"user"/"assistant", "content": str}]
            The conversation history before the current assistant reply.
        answer: str
            The current assistant reply to supervise.
        max_len: int
            Maximum total token length.

    Returns:
        dict containing:
            input_ids, attention_mask, labels, prompt_str, label_text
    """

    # Build the prompt up to the assistant header, without the target answer.
    prompt_str = tokenizer.apply_chat_template(
        context_messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Encode prompt and answer separately.
    prompt_enc = tokenizer(prompt_str, add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"]
    prompt_mask = prompt_enc["attention_mask"]

    answer_text = answer + EOT_TOKEN
    answer_enc = tokenizer(answer_text, add_special_tokens=False)
    answer_ids = answer_enc["input_ids"]
    answer_mask = answer_enc["attention_mask"]

    # Strictly enforce total length <= max_len.
    total_len = len(prompt_ids) + len(answer_ids)
    if total_len > max_len:
        # If the answer itself is too long, keep only its tail.
        if len(answer_ids) >= max_len:
            answer_ids = answer_ids[-max_len:]
            answer_mask = answer_mask[-max_len:]
            prompt_ids, prompt_mask = [], []
        else:
            # Otherwise truncate the prompt and keep the part closest to the answer.
            keep_prompt = max_len - len(answer_ids)
            if keep_prompt <= 0:
                prompt_ids, prompt_mask = [], []
            else:
                prompt_ids = prompt_ids[-keep_prompt:]
                prompt_mask = prompt_mask[-keep_prompt:]

    input_ids = prompt_ids + answer_ids
    attention_mask = prompt_mask + answer_mask
    labels = [-100] * len(prompt_ids) + answer_ids

    assert len(input_ids) == len(attention_mask) == len(labels)
    assert len(input_ids) <= max_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_str": prompt_str,
        "label_text": answer,
    }


def expand_conversation_to_sft_samples(conv_sample, max_len=1000):
    """
    Expand one multi-turn conversation into multiple SFT samples.

    Args:
        conv_sample: dict with at least:
            - "messages": list[{"role": "user"/"assistant"/"system", "content": str}]
            - optional extra fields such as "id"
        max_len: int

    Returns:
        list[dict], where each dict is a directly usable SFT sample.
    """
    messages = conv_sample["messages"]
    sft_samples = []

    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        answer = msg["content"]
        context_messages = messages[:idx]

        # Skip dirty samples with no user message in the context.
        if not any(m["role"] == "user" for m in context_messages):
            continue

        features = build_input_with_context(
            context_messages=context_messages,
            answer=answer,
            max_len=max_len,
        )

        features["conv_id"] = conv_sample.get("id", None)
        features["turn_index"] = idx
        sft_samples.append(features)

    return sft_samples


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {line_no}: {e}") from e
    return data


def map_batch(batch, max_len=3000):
    """
    Expand a batch of multi-turn conversations into SFT samples.

    Args:
        batch: batch from Hugging Face Dataset
        max_len: maximum sequence length during construction

    Returns:
        dict of column-wise lists
    """
    batch_messages = batch["messages"]
    batch_ids = batch.get("id", [None] * len(batch_messages))

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_conv_id = []
    all_turn_index = []
    all_prompt_str = []
    all_label_text = []

    for msgs, cid in zip(batch_messages, batch_ids):
        conv_sample = {"messages": msgs, "id": cid}
        sft_samples = expand_conversation_to_sft_samples(conv_sample, max_len=max_len)

        for sample in sft_samples:
            all_input_ids.append(sample["input_ids"])
            all_attention_mask.append(sample["attention_mask"])
            all_labels.append(sample["labels"])
            all_conv_id.append(sample["conv_id"])
            all_turn_index.append(sample["turn_index"])
            all_prompt_str.append(sample["prompt_str"])
            all_label_text.append(sample["label_text"])

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
        "conv_id": all_conv_id,
        "turn_index": all_turn_index,
        "prompt_str": all_prompt_str,
        "label_text": all_label_text,
    }


def filter_by_max_len(ds: Dataset, max_len: int = 2000) -> Dataset:
    """Keep only samples whose input length is smaller than max_len."""

    def _filter(example):
        return len(example["input_ids"]) < max_len

    return ds.filter(_filter)


def process_and_save_dataset(input_path, split, output_root, build_max_len=3000, filter_max_len=2000):
    """
    Process one JSONL file and save the unrolled/tokenized dataset to disk.

    Args:
        input_path: path to the source JSONL file
        split: "train" or "eval"
        output_root: root output directory
        build_max_len: max length during sample construction
        filter_max_len: final max length filter
    """
    input_path = Path(input_path)
    save_dir = Path(output_root) / split / input_path.stem
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing [{split}] {input_path}")
    raw_data = read_jsonl(input_path)
    print(f"Loaded {len(raw_data)} conversations.")

    data_all = Dataset.from_list(raw_data)

    tokenized_ds = data_all.map(
        lambda batch: map_batch(batch, max_len=build_max_len),
        batched=True,
        remove_columns=data_all.column_names,
        desc=f"Tokenizing and unrolling: {input_path.name}",
    )

    filtered_ds = filter_by_max_len(tokenized_ds, filter_max_len)
    filtered_ds.save_to_disk(str(save_dir))

    print(f"Saved to: {save_dir}")
    print(f"Final sample count: {len(filtered_ds)}")


def main():
    for train_path in TRAIN_INPUT_PATHS:
        process_and_save_dataset(
            input_path=train_path,
            split="train",
            output_root=OUTPUT_ROOT,
            build_max_len=MAX_BUILD_LEN,
            filter_max_len=MAX_FILTER_LEN,
        )

    for eval_path in EVAL_INPUT_PATHS:
        process_and_save_dataset(
            input_path=eval_path,
            split="eval",
            output_root=OUTPUT_ROOT,
            build_max_len=MAX_BUILD_LEN,
            filter_max_len=MAX_FILTER_LEN,
        )

    elapsed = time.time() - start_time
    print(f"\nAll datasets have been processed. Total time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
