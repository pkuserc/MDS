<div align="center">

# 面向多轮对话指令微调的数据筛选方法

<p>
  <a href="https://github.com/WisdomShell/MDS">English</a> | <strong>简体中文</strong>
</p>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.07892)
[![Venue](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Task](https://img.shields.io/badge/Task-Dialogue%20Data%20Selection-purple.svg)](#概述)

**ACL 2026 Findings**

<a href="https://deepblue666.github.io/">Bo Li</a>, Shikun Zhang, Wei Ye

</div>

本仓库公开了我们在多轮对话指令微调工作（MDS）中使用的核心数据选择与评测流程。当前公开内容包括：

- MDS 全局阶段选择代码
- MDS 局部阶段选择代码
- 将选出的训练/测试对话转换为 disk-style 数据的脚本
- 多轮对话预测结果的统一评测代码
- 公开的训练与评测数据集
- 记录依赖包及版本的环境文件 `requirements.txt`

---

## 概述

多轮对话语料往往存在噪声大、结构不一致等问题。与单轮指令数据相比，它更容易出现话题漂移、重复闲聊、信息推进不足，以及多轮之间回答格式不匹配等现象。MDS 从**对话级数据选择**的角度来解决这一问题。

MDS 包含两个阶段：

1. **全局阶段**  
   MDS 首先在**用户查询轨迹空间**中表示每个对话，将对话聚类到语义 bin 中，并在每个 bin 内进行带冗余控制的候选选择。该阶段旨在尽量保留广泛的语义覆盖，同时避免高频交互模式被过度选中。

2. **局部阶段**  
   MDS 随后通过**对话级结构打分**进一步精炼全局候选池，其中包括基于实体的信号以及问答形式/风格一致性信号。最终训练子集通过带预算约束的 bin-wise 方式选出。

论文中的完整流程为：

`dialogue pool -> global-stage candidate construction -> local-stage selection -> disk-style train/eval data -> model training -> model prediction -> evaluation`

本仓库主要覆盖其中的**数据选择、数据转换和评测**部分。

---

## 公开内容

本仓库包含以下组成部分：

### 代码
- `mds_global_stage.py`  
  在用户查询轨迹空间中进行全局阶段的对话选择。

- `mds_local_stage.py`  
  进行局部阶段的对话打分与最终按 bin 预算选择。

- `build_disk_data.py`  
  将选出的训练对话和公开的评测对话转换为下游训练或评测所需的 disk-style 数据格式。

- `mds_dialogue_eval.py`  
  用于模型预测文件的统一评测脚本。

### 环境
- `requirements.txt`  
  列出本仓库所需依赖包及版本的环境文件。

### 数据
- `data/train/`
  - `baize_chat_data_use.jsonl`
  - `banking_train_50k_use.jsonl`

- `data/eval/`
  - `banking_test_300.jsonl`
  - `ConsistentChat_use_270.jsonl`
  - `dialogue_test_unseen_use.jsonl`
  - `mt_eval_part_use.jsonl`

---

## 仓库结构

```text
.
├── README.md
├── README_zh.md
├── requirements.txt
├── mds_global_stage.py
├── mds_local_stage.py
├── build_disk_data.py
├── mds_dialogue_eval.py
├── data/
│   ├── train/
│   │   ├── baize_chat_data_use.jsonl
│   │   └── banking_train_50k_use.jsonl
│   └── eval/
│       ├── banking_test_300.jsonl
│       ├── ConsistentChat_use_270.jsonl
│       ├── dialogue_test_unseen_use.jsonl
│       └── mt_eval_part_use.jsonl
└── outputs/
```

典型流程如下：

```text
data/train/*.jsonl
    -> mds_global_stage.py
    -> outputs/mds_global_candidates*.jsonl
    -> mds_local_stage.py
    -> outputs/mds_final_selected*.jsonl
    -> build_disk_data.py
    -> disk-style train / eval data
    -> model training (external to this repo)
    -> model inference (external to this repo)
    -> mds_dialogue_eval.py
```

---

## 安装

建议使用 Python 3.10 及以上版本。

可按你喜欢的方式安装环境，例如：

```bash
pip install -r requirements.txt
```

根据你的运行环境，可能还需要额外安装 GPU 相关依赖。

---

## 数据组织

### 训练数据

公开的训练对话数据位于：

```text
data/train/
```

当前训练文件：
- `baize_chat_data_use.jsonl`
- `banking_train_50k_use.jsonl`

这些文件用于运行 MDS 的数据选择流程。

### 评测数据

公开的评测数据位于：

```text
data/eval/
```

当前评测文件：
- `banking_test_300.jsonl`
- `ConsistentChat_use_270.jsonl`
- `dialogue_test_unseen_use.jsonl`
- `mt_eval_part_use.jsonl`

这些文件用于评测，或转换为 disk-style 的评测数据。

### 对话格式

每一行 JSONL 表示一个对话级样本。  
本仓库默认推荐的数据格式如下：

```json
{
  "messages": [
    {"content": "I want to know the step by step guide to invest in share market in India.", "role": "user"},
    {"content": "Sure, I can help with that. Firstly, you need to open a demat and trading account with a registered stockbroker.", "role": "assistant"},
    {"content": "How do I find a registered stockbroker in India?", "role": "user"},
    {"content": "You can visit the websites of National Stock Exchange (NSE) or Bombay Stock Exchange (BSE) to get a list of registered stockbrokers in India.", "role": "assistant"},
    {"content": "What documents are required to open a demat and trading account?", "role": "user"},
    {"content": "You will need to provide identity proof (PAN card), address proof, bank details and a passport size photograph to open a demat and trading account.", "role": "assistant"},
    {"content": "How do I start trading once I have a demat and trading account?", "role": "user"},
    {"content": "You can start trading by placing buy and sell orders for stocks through your stockbroker either online or offline.", "role": "assistant"},
    {"content": "How do I track my investments?", "role": "user"},
    {"content": "You can track your investments through your demat account. It will provide you with a consolidated view of all your investments in various stocks and other financial instruments.", "role": "assistant"}
  ],
  "id": "conv_id_0"
}
```

要求：
- `messages` 必须是对话轮次列表
- 每一轮都必须包含 `role` 和 `content`
- `role` 通常为 `user` 或 `assistant`
- `id` 应唯一标识该对话

公开脚本也可能支持一些相近变体，但以上格式是推荐默认格式。

---

## 支持的输入格式

### 1. 对话池格式

训练对话文件应为 JSONL 格式，每行一个对话。  
本仓库主要使用的格式为：

```json
{
  "messages": [
    {"role": "user", "content": "User utterance 1"},
    {"role": "assistant", "content": "Assistant reply 1"},
    {"role": "user", "content": "User utterance 2"},
    {"role": "assistant", "content": "Assistant reply 2"}
  ],
  "id": "conv_id_xxx"
}
```

### 2. 全局 / 局部选择输出格式

选择脚本会保留原始对话内容，并可能附加如下元数据：
- `conv_id`
- `bin_id` / `cluster_id`
- 与选择相关的中间字段
- 局部阶段分数

具体字段取决于脚本配置和中间处理步骤。

### 3. 评测输入格式

评测脚本接收 JSONL 格式的预测文件。  
每一行至少应包含：

```json
{
  "idx": 0,
  "conv_id": "dialog_001",
  "turn_index": 3,
  "prompt": "<prompt string>",
  "gold": "reference answer",
  "prediction": "model output"
}
```

评测脚本会把各类指标追加到每条记录中，也可以保存汇总统计信息。

---

## 方法流程

### 第一步：全局阶段候选构建

全局阶段会从用户轮次构建对话表示，将对话聚类到语义 bin 中，并在每个 bin 内进行覆盖性与冗余控制下的候选选择。

示例：

```bash
python mds_global_stage.py \
  --input_jsonl data/train/baize_chat_data_use.jsonl \
  --output_jsonl outputs/mds_global_candidates.jsonl \
  --output_ids_json outputs/mds_global_candidate_ids.json \
  --n_clusters 1000 \
  --retain_ratio 0.5 \
  --min_keep_per_cluster 5 \
  --lambda_mmr 0.5
```

主要输出：
- 候选对话 JSONL
- JSON 格式的已选对话 ID 列表

---

### 第二步：局部阶段最终选择

局部阶段通过对话级结构打分进一步精炼候选池，并执行带预算约束的最终选择。

示例：

```bash
python mds_local_stage.py \
  --input_jsonl outputs/mds_global_candidates.jsonl \
  --output_jsonl outputs/mds_final_selected.jsonl \
  --output_ids_json outputs/mds_final_selected_ids.json \
  --judge_model /path/to/judge_model \
  --budget 10000 \
  --tau_form 1.0
```

主要输出：
- 最终选中的对话 JSONL
- JSON 格式的最终对话 ID 列表

---

### 第三步：将训练/评测数据转换为 Disk-Style 格式

转换脚本会将多轮对话展开为逐轮样本，并以 disk-style 格式保存，供下游训练或评测使用。

示例：

```bash
python build_disk_data.py
```

运行前请在脚本中更新以下路径：
- 模型路径
- 输入 JSONL 文件路径
- 输出根目录

默认情况下，转换脚本会在配置的输出根目录下保存：
- `train/` 下的训练集
- `eval/` 下的评测集

---

### 第四步：模型训练

完成对话选择后，选中的对话会被展开成 turn-level 的监督样本，并用于标准 chat-style 监督微调。你可以使用常见的 SFT / LoRA 训练框架完成这一步。

一个典型训练流程如下：

1. 在原始对话池上运行 MDS 全局阶段选择
2. 运行 MDS 局部阶段选择，获得最终子集
3. 将选出的对话转换为 disk-style 训练数据
4. 在处理后的逐轮样本上微调 backbone chat 模型
5. 在评测集上运行推理
6. 使用 `mds_dialogue_eval.py` 对预测结果进行评测

---

### 第五步：运行模型推理

你可以使用任意标准 chat 推理流程，对转换后的评测数据生成预测结果。

最终预测文件应符合前文给出的评测 JSONL 格式。

---

### 第六步：统一评测

评测脚本为多种指标提供统一接口。

示例：

```bash
python mds_dialogue_eval.py
```

运行前请更新脚本中的相关设置，例如：
- OpenAI-compatible API 配置
- 评测输入文件路径
- 输出文件路径
- 启用 / 关闭的指标开关
- 如有需要的 embedding 模型路径

---

## 评测指标

统一评测脚本支持以下指标：

### GPT 类指标
- **LLM-EVAL**
  - content
  - grammar
  - relevance
  - appropriateness
  - overall

- **G-EVAL**
  - coherence
  - naturalness
  - engagement
  - groundedness
  - overall

- **Entity-F1**
  - 基于 GPT 的实体抽取
  - 抽取实体上的 micro F1

### 本地 / 非 GPT 指标
- **Embedding cosine similarity**
  - 基于 sentence-transformers

- **BLEU-3**
  - 基于 sacreBLEU

- **ROUGE-L**
  - 基于 rouge-score

### 说明
- GPT 类指标需要有效的 API 配置。
- 在较大的测试集上，受模型接口和速率限制影响，GPT 类评测可能耗时较长。
- 不同 API 后端或 judge 模型可能会带来轻微分数波动。

---

## 快速开始

一个最小可运行的端到端流程如下：

```bash
# 1. 全局阶段选择
python mds_global_stage.py \
  --input_jsonl data/train/baize_chat_data_use.jsonl \
  --output_jsonl outputs/mds_global_candidates.jsonl \
  --output_ids_json outputs/mds_global_candidate_ids.json

# 2. 局部阶段选择
python mds_local_stage.py \
  --input_jsonl outputs/mds_global_candidates.jsonl \
  --output_jsonl outputs/mds_final_selected.jsonl \
  --output_ids_json outputs/mds_final_selected_ids.json \
  --judge_model /path/to/judge_model \
  --budget 10000 \
  --tau_form 1.0

# 3. 转换选中训练/评测数据为 disk 格式
python build_disk_data.py

# 4. 使用任意标准 SFT / LoRA 框架训练模型

# 5. 在评测数据上运行推理

# 6. 评测预测文件
python mds_dialogue_eval.py
```

---

## 脚本说明

### `mds_global_stage.py`
主要功能：
- 读取对话级 JSONL 文件
- 提取用户查询轨迹表示
- 计算对话嵌入
- 将对话聚类到语义 bin 中
- 执行 bin-wise 的 MMR 风格候选选择

### `mds_local_stage.py`
主要功能：
- 规范化对话格式
- 从对话中抽取 QA 对
- 使用 judge model 做逐轮形式/风格分析
- 计算局部结构信号
- 执行按 bin 预算约束的最终选择

### `build_disk_data.py`
主要功能：
- 读取选中的训练对话和评测对话文件
- 将每个对话展开为 turn-level 的监督样本
- 对 prompt 和 label 进行 tokenize
- 以 disk-style 格式保存处理后的数据集

### `mds_dialogue_eval.py`
主要功能：
- 读取预测 JSONL 文件
- 计算 GPT 类和本地自动指标
- 保存逐样本结果
- 保存或打印汇总结果

---

## 常见问题

### 1. 评测时 API 超时
在大 batch 或严格速率限制下，GPT 类评测可能出现超时。

建议处理方式：
- 减少 worker 数量
- 降低请求并发
- 重试失败样本
- 适当增大超时设置

### 2. 转换脚本中缺少模型路径
运行前请先在转换脚本中更新 tokenizer / model 路径。

### 3. 输入格式不匹配
如果你的对话 JSONL 不符合推荐的 `messages` + `role` + `content` + `id` 格式，可能需要在运行流程前增加一个轻量预处理步骤。

### 4. Judge model 不一致
为了更稳定地复现结果，建议在不同运行中固定局部阶段使用的 judge model。

### 5. 仓库未包含训练与预测代码
这是预期行为。本仓库主要公开与选择和评测直接相关的部分，训练和预测阶段可使用标准 chat 微调框架复现。

---

## 引用

```bibtex
@misc{li2026dataselectionmultiturndialogue,
  title         = {Data Selection for Multi-turn Dialogue Instruction Tuning},
  author        = {Bo Li and Shikun Zhang and Wei Ye},
  year          = {2026},
  eprint        = {2604.07892},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.07892}
}
```
