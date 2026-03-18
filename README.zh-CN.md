# ACF-Stego

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#1-环境安装)

该仓库是论文 **Asymmetric Collaborative Framework (ACF)**（认知非对称场景）的官方代码实现。

## 目录
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [与论文对齐的设置](#与论文对齐的设置)
- [协议映射](#协议映射)
- [实验输出](#实验输出)

## 项目结构

```text
ACF-Stego/
├── config/                          # 运行、数据集、模型与实验配置
│   ├── runtime.py                   # API/运行时环境变量
│   ├── dataset.py                   # 数据集来源与缓存设置
│   ├── experiment.py                # 实验阶段与超参数
│   ├── models.py                    # 模型别名 -> 模型路径/ID
│   └── paths.py                     # 项目路径常量
├── core/tools/                      # LongMemEval 工具、检索与评估
├── experiments/                     # 主实验入口（group1...group8, controlled）
├── scripts/
│   ├── run_v2_pipeline.sh           # 一键实验流水线
│   ├── score_v2_llm_judge.py        # realistic 语义 LLM 评测
│   └── analyze_v2_outputs.py        # 生成两张表 + 两张图
├── tests/                           # 单元测试
├── .env.example                     # 环境变量模板
├── requirements.txt                 # Python 依赖
├── LICENSE                          # MIT 许可证
├── README.md                        # 英文 README
└── README.zh-CN.md                  # 中文 README
```

## 快速开始

### 1. 环境安装

```bash
conda create -n acf-stego python=3.10 -y
conda activate acf-stego
pip install -r requirements.txt
cp .env.example .env
```

在调用远程 API 前，请在 `.env` 中填写你自己的密钥：
- `OPENAI_API_KEY`
- `HF_TOKEN`

### 2. 数据集准备

LongMemEval-S（cleaned）官方下载链接：
- https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json

下载并放到以下路径：

```bash
mkdir -p data/raw
wget -O data/raw/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json"
```

期望路径：
- `data/raw/longmemeval_s_cleaned.json`

### 3. 运行主要实验

```bash
bash scripts/run_v2_pipeline.sh --stage recommended
```

该命令会执行：
- controlled 套件（`v2_controlled_asymmetry`）
- realistic 套件（`group1` ... `group8`）
- realistic 语义 LLM 评测
- 聚合并生成图和表

常用替代命令：

```bash
# 快速调试
bash scripts/run_v2_pipeline.sh --stage debug --judge-limit 20 --analysis-suffix debug

# 只重跑聚合产物
python scripts/analyze_v2_outputs.py --experiment all
```

## 与论文对齐的设置

当前仓库中的论文结果默认对应以下设置：
- 数据集切分：`longmemeval_s`（cleaned）
- 基础模型：`Qwen/Qwen2.5-7B-Instruct`（`MODEL_NAME=QWEN2_5_7B_INSTRUCT`）
- 公共上下文窗口：`LONGMEMEVAL_WINDOW_SESSIONS=5`
- controlled 截断不匹配：`LONGMEMEVAL_DRIFT_KEEP_SESSIONS=3`（`drift_recent3`）
- controlled sweep 强度：`LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS=4,3,2,1`
- ACF 设置：`LONGMEMEVAL_ACF_K_VALUES=8,12,16`
- 语义评分模型（0/1/2）：`LLM_JUDGE_MODEL=gemini-2.0-flash`

## 协议映射

脚本中的 `group*` 与论文协议名对应关系如下：

| 脚本分组 | 论文协议名 |
| --- | --- |
| `group1` | `Normal (No Stego)` |
| `group2` | `DISCOP` |
| `group3` | `METEOR` |
| `group4` | `ACF` |
| `group5` | `ACF+RET` |
| `group6` | `DISCOP+RET` |
| `group7` | `METEOR+RET` |
| `group8` | `Normal+RET` |

## 实验输出

主要输出目录：
- `data/outputs_v2/controlled/`
- `data/outputs_v2/controlled_sweep/`
- `data/outputs_v2/controlled_summary/`
- `data/outputs_v2/realistic/`
- `data/table/v2/`

表（`data/table/v2/`）：
- `paper_table_controlled_cognitive_asymmetry.csv`
- `paper_table_realistic_integrated.csv`

图（`data/table/v2/figures/`）：
- `figure_ber_vs_decoder_sessions.pdf`
- `figure_tradeoff_grouped_bar.pdf`
