# EXAONE 경량화 해커톤 가이드

> **2026 LG Aimers 8기** 해커톤 문제 소개 강의자료 정리

---

## 목차

1. [해커톤 배경 및 목표](#1-해커톤-배경-및-목표)
2. [EXAONE 4.0 구조 분석](#2-exaone-40-구조-분석)
3. [경량화 적용 - LLM Compressor](#3-경량화-적용---llm-compressor)
4. [추론 엔진 적용 - vLLM](#4-추론-엔진-적용---vllm)
5. [평가 지표](#5-평가-지표)
6. [부록 - OpenAI Compatible](#6-부록---openai-compatible)

---

## 1. 해커톤 배경 및 목표

### 1.1 배경

- EXAONE은 **Global Frontier급 Large-scale 모델**과 **On-Device를 지원하기 위한 Small-scale 모델**이 있음
- 랩탑을 위한 **2.4B**, 스마트폰을 위한 **1.2B** 모델이 있으나 **더 작고 빠른 모델에 대한 요구사항**이 있음
- 단순히 파라미터 수를 더 줄이면 메모리와 속도 요건은 만족하나 **정확도가 크게 열화됨**
- **모델 크기를 줄이고 빠르게 하면서도 정확도를 유지**할 수 있는 경량화 기법을 모색하고자 과제를 제안함

### 1.2 경량화 단계

```
EXAONE-4.0 분석 → 경량화 적용 → 추론 엔진 적용 → 평가
```

### 1.3 기대 효과

- **On-Device 환경**에서 원활히 구동할 수 있는 EXAONE 모델 지원
- **Large-scale EXAONE 모델**에도 확대 적용하여 전체 서비스의 운영 비용 감소

---

## 2. EXAONE 4.0 구조 분석

### 2.1 모델 다운로드

- **허깅페이스**: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/tree/main
- `config.json` 파일에서 모델의 상세한 구조 정보를 얻을 수 있음

### 2.2 모델 구조 비교 (32B vs 1.2B)

| 항목 | 32B | 1.2B |
|------|-----|------|
| d_model | 5,120 | 2,048 |
| Number of layers | 64 | 30 |
| Normalization | QK-Reorder-LN | QK-Reorder-LN |
| Non-linearity | SwiGLU [50] | SwiGLU |
| Feedforward dimension | 27,392 | 4,096 |
| Attention type | **Hybrid** | **Global** |
| Head type | GQA [4] | GQA |
| Number of heads | 40 | 32 |
| Number of KV heads | 8 | 8 |
| Head size | 128 | 64 |
| Max sequence length | 131,072 | 65,536 |
| RoPE theta [52] | 1,000,000 | 1,000,000 |
| Tokenizer | BBPE [58] | BBPE |
| Vocab size | 102,400 | 102,400 |
| Tied word embedding | **False** | **True** |
| Knowledge cut-off | Nov. 2024 | Nov. 2024 |

### 2.3 EXAONE-4.0의 구조적 특징

#### 2.3.1 Sliding Window Hybrid Attention (32B)

- **3:1 비율**로 Local (Sliding Window) Attention과 Global Attention을 Hybrid로 적용
- **Local Attention**을 적용해 Attention 연산을 줄이고 추론시 **KV Cache Memory를 절감**
- **Global Attention**을 Hybrid로 사용해 열화되는 정확도를 보존

```
Layer 구성 예시 (Window Size: 6):
- Global: 전체 컨텍스트 참조
- Local: Window Size 내에서만 참조 (Sliding)
- Local: Window Size 내에서만 참조
- Local: Window Size 내에서만 참조
... (3:1 비율로 반복)
```

#### 2.3.2 QK-Reorder-LN

- LayerNorm의 위치를 변경하고 **Query, Key Projection에 LayerNorm을 추가**
- 약간의 연산량 추가로 **더 높은 성능**을 달성할 수 있음

**EXAONE 4.0 (QK-Reorder-LN) 구조:**
```
Input
  ↓
RMS Norm
  ↓
Attention Block
  ├── Query → RMS Norm
  ├── Key → RMS Norm
  └── Value
  ↓
Attention Weights → Softmax → Attention Output
  ↓
(+) Residual Connection
  ↓
RMS Norm
  ↓
FFN Block
  ↓
(+) Residual Connection
  ↓
Output
```

**EXAONE 3.5 (Pre-LN) 구조:**
```
Input
  ↓
RMS Norm
  ↓
Attention Block
  ↓
(+) Residual Connection
  ↓
RMS Norm
  ↓
FFN Block
  ↓
(+) Residual Connection
  ↓
Output
```

### 2.4 모델 파일 정보

**model.safetensors 구조 (1.2B 모델):**

| Tensor | Shape | Precision |
|--------|-------|-----------|
| model.embed_tokens.weight | [102,400, 2,048] | BF16 |
| model.layers.0.mlp.down_proj.weight | [2,048, 4,096] | BF16 |
| model.layers.0.mlp.gate_proj.weight | [4,096, 2,048] | BF16 |
| model.layers.0.mlp.up_proj.weight | [4,096, 2,048] | BF16 |
| model.layers.0.post_attention_layernorm.weight | [2,048] | BF16 |
| model.layers.0.post_feedforward_layernorm.weight | [2,048] | BF16 |
| model.layers.0.self_attn.k_norm.weight | [64] | BF16 |
| model.layers.0.self_attn.k_proj.weight | [512, 2,048] | BF16 |
| model.layers.0.self_attn.o_proj.weight | [2,048, 2,048] | BF16 |
| model.layers.0.self_attn.q_norm.weight | [64] | BF16 |
| model.layers.0.self_attn.q_proj.weight | [2,048, 2,048] | BF16 |
| model.layers.0.self_attn.v_proj.weight | [512, 2,048] | BF16 |

---

## 3. 경량화 적용 - LLM Compressor

### 3.1 기본 사용법

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)}

ds = ds.map(preprocess)
```

### 3.2 양자화 설정

```python
# Configure the quantization algorithm to run.
recipe = [ GPTQModifier(ignore=["embed_tokens", "lm_head"], scheme="W4A16", targets=["Linear"]) ]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
message = [{"role": "user", "content": "Who are you?"}]
input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3.3 GPTQModifier 주요 인자

| 인자 | 설명 |
|------|------|
| `ignore` | 양자화를 **제외**할 모듈을 지정 |
| `scheme` | Weight와 Activation을 어떤 precision으로 사용할지 지정 |
| `targets` | ignore와 반대로 양자화를 **할** 모듈을 지정 |

### 3.4 참고 자료

- Compression Schemes: https://github.com/vllm-project/llm-compressor/blob/main/docs/guides/compression_schemes.md
- Examples: https://github.com/vllm-project/llm-compressor/tree/main/examples
- GPTQModel: https://github.com/ModelCloud/GPTQModel/tree/main?tab=readme-ov-file#quantization-support

### 3.5 최신 모델 경향

#### OpenAI GPT-oss-120b

```json
"quantization_config": {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head"
    ],
    "quant_method": "mxfp4"
}
```

#### Moonshot Kimi-K2-Thinking

```json
"quantization_config": {
    "config_groups": {
        "group_0": {
            "input_activations": null,
            "output_activations": null,
            "targets": ["Linear"],
            "weights": {
                "actorder": null,
                "block_structure": null,
                "dynamic": false,
                "group_size": 32,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": true,
                "type": "int"
            }
        }
    },
    "format": "pack-quantized",
    "ignore": [
        "lm_head",
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
}
```

---

## 4. 추론 엔진 적용 - vLLM

### 4.1 기본 사용법

```python
from vllm import LLM, SamplingParams

prompts = [
    [{"role": "user", "content": "Explain how wonderful you are"}],
    [{"role": "user", "content": "너가 얼마나 대단한지 설명해 봐"}],
]
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

llm = LLM(model="EXAONE-4.0-1.2B-GPTQ")

outputs = llm.chat(prompts, sampling_params)

for output in outputs:
    print("############")
    print(output.outputs[0].text)
    print()
```

**실행:**
```bash
python3 vllm_inference.py
```

### 4.2 vLLM 지원 Quantization 기법

| Implementation | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | Intel Gaudi | x86 CPU | Google TPU |
|---------------|-------|--------|--------|-----|--------|---------|-----------|-------------|---------|------------|
| AWQ | X | O | O | O | O | X | O | X | O | X |
| GPTQ | O | O | O | O | O | X | O | X | O | X |
| Marlin (GPTQ/AWQ/FP8) | X | X | O | O | O | X | X | X | X | X |
| INT8 (W8A8) | X | O | O | O | O | X | X | X | O | O |
| FP8 (W8A8) | X | X | X | O | O | X | X | X | X | X |
| BitBLAS | O | O | O | O | O | X | X | X | X | X |
| BitBLAS (GPTQ) | X | X | O | O | O | X | X | X | X | X |
| bitsandbytes | O | O | O | O | O | X | X | X | X | X |
| DeepSpeedFP | O | O | O | O | O | X | X | X | X | X |
| GGUF | O | O | O | O | O | O | X | X | X | X |
| INC (W8A8) | X | X | X | X | X | X | X | O | X | X |

### 4.3 GPU 아키텍처별 GPU 종류

| 아키텍처 | GPU 종류 |
|---------|---------|
| **Volta** | V100 등 |
| **Turing** | T4, GeForce RTX 20 시리즈 등 |
| **Ampere** | A100, A10 등 |
| **Ada Lovelace** | GeForce RTX 40 시리즈, L4 등 |
| **Hopper** | H100, H800 등 |

### 4.4 참고 자료

- vLLM Quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart
- vLLM Quantization: https://github.com/vllm-project/vllm/tree/main/docs/features/quantization

---

## 5. 평가 지표

### 5.1 Accuracy 평가

#### 벤치마크 데이터셋 (Small-size 모델 비교)

**Reasoning 모드:**

| 벤치마크 | EXAONE 4.0 1.2B (Reasoning) | EXAONE Deep 2.4B (Reasoning) | Qwen 3 0.6B (Reasoning) | Qwen 3 1.7B (Reasoning) | SmolLM 3 3B (Reasoning) |
|---------|---------------------------|----------------------------|------------------------|------------------------|------------------------|
| **Type** | Hybrid 1.28B | Reasoning 2.41B | Hybrid 596M | Hybrid 1.72B | Hybrid 3.08B |
| **World Knowledge** |
| MMLU-Redux | 71.5 | 68.9 | 55.6* | 73.9* | 74.8 |
| MMLU-Pro | 59.3 | 56.4* | 38.3 | 57.7 | 57.8 |
| GPQA-Diamond | 52.0 | 54.3* | 27.9* | 40.1* | 41.7* |
| **Math / Coding** |
| AIME 2025 | 45.2 | 47.9* | 15.1* | 36.8* | 36.7* |
| HMMT Feb 2025 | 34.0 | 27.3 | 7.0 | 21.8 | 26.0 |
| LiveCodeBench v5 | 44.6 | 47.2 | 12.3* | 33.2* | 27.6 |
| LiveCodeBench v6 | 45.3 | 43.1 | 16.4 | 29.9 | 29.1 |
| **Instruction Following** |
| IFEval | 67.8 | 71.0 | 59.2* | 72.5* | 71.2* |
| Multi-IF (EN) | 53.9 | 54.5 | 37.5 | 53.5 | 47.5 |
| **Agentic Tool Use** |
| BFCL-v3 | 52.9 | N/A | 46.4* | 56.4* | 37.1 |
| Tau-Bench (Airline) | 20.5 | N/A | 22.0 | 31.0 | 37.0 |
| Tau-Bench (Retail) | 28.1 | N/A | 3.3 | 6.5 | 5.4 |
| **Multilinguality** |
| KMMLU-Pro (KO) | 42.7 | 24.6 | 21.6 | 38.3 | 30.5 |
| KMMLU-Redux (KO) | 46.9 | 25.0 | 24.5 | 38.0 | 33.7 |
| KSM (KO) | 60.6 | 60.9 | 22.8 | 52.9 | 49.7 |
| MMMLU (ES) | 62.4 | 51.4 | 48.8* | 64.5* | 64.7 |
| MATH500 (ES) | 88.8 | 84.5 | 70.6 | 87.9 | 87.5 |

### 5.2 lm-evaluation-harness 사용법

```bash
MODEL_ID=EXAONE-4.0-1.2B-GPTQ

lm_eval --model vllm \
    --model_args pretrained=${MODEL_ID},gpu_memory_utilization=0.85,enable_thinking=False,max_gen_toks=2048 \
    --tasks gsm8k \
    --limit 512 \
    --output_path results \
    --apply_chat_template \
    --batch_size auto
```

**실행:**
```bash
bash run_lmeval.sh
```

**결과 비교:**

| 모델 | Tasks | Version | Filter | n-shot | Metric | Value | Stderr |
|------|-------|---------|--------|--------|--------|-------|--------|
| EXAONE-4.0-1.2B | gsm8k | 3 | flexible-extract | 5 | exact_match↑ | 0.6484 | ±0.0211 |
| EXAONE-4.0-1.2B | gsm8k | 3 | strict-match | 5 | exact_match↑ | 0.5645 | ±0.0219 |
| **Quantized** EXAONE-4.0-1.2B | gsm8k | 3 | flexible-extract | 5 | exact_match↑ | 0.5977 | ±0.0217 |
| **Quantized** EXAONE-4.0-1.2B | gsm8k | 3 | strict-match | 5 | exact_match↑ | 0.4727 | ±0.0221 |

**참고:** https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks#tasks

### 5.3 Memory 평가

최종적으로 저장되는 **safetensors 파일의 크기**를 측정

| 모델 | model.safetensors 크기 |
|------|----------------------|
| EXAONE-4.0-1.2B (원본) | 2,558,821,288 bytes (~2.56 GB) |
| EXAONE-4.0-1.2B-GPTQ (양자화) | 1,390,692,528 bytes (~1.39 GB) |

**압축률:** 약 **46% 감소**

---

## 6. 부록 - OpenAI Compatible

### 6.1 개요

- 최근에는 vLLM과 같은 추론엔진을 **OpenAI Compatible Server** 형태로 구동하고 평가 프레임워크에서 API를 호출하는 형태의 평가 방식이 인기
- 개발자들 사이에서 OpenAI 라이브러리가 대중화되고 어떤 평가든 일관된 포맷으로 평가가 가능해 쉽게 구현 및 구동이 쉽다는 장점이 있음
- **OpenAI Compatible**은 오픈소스계에서 최소 조건이 되어가고 있음

### 6.2 vLLM OpenAI Compatible Server

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
    --dtype auto
```

**참고:** https://docs.vllm.ai/en/latest/serving/openai_compatible_server/

### 6.3 평가 프레임워크

#### OpenAI Simple Evals

```bash
# 사용 가능한 모델 확인
python -m simple-evals.simple_evals --list-models

# 평가 실행
python -m simple-evals.simple_evals --model <model_name> --examples <num_examples>
```

**참고:** https://github.com/openai/simple-evals

#### NVIDIA NeMo Evaluator

```bash
# Launcher 설치
pip install nemo-evaluator-launcher
```

**참고:** https://github.com/NVIDIA-NeMo/Evaluator

---

## 참고 링크 모음

### 모델 & 데이터셋
- EXAONE-4.0-1.2B: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/tree/main
- MANTA-1M Dataset: https://huggingface.co/LGAI-EXAONE/MANTA-1M
- EXAONE-4.0 Technical Report: https://arxiv.org/abs/2507.11407

### 경량화 도구
- LLM Compressor: https://github.com/vllm-project/llm-compressor
- GPTQModel: https://github.com/ModelCloud/GPTQModel

### 추론 엔진
- vLLM: https://docs.vllm.ai/en/latest/getting_started/quickstart
- vLLM Quantization: https://github.com/vllm-project/vllm/tree/main/docs/features/quantization

### 평가 프레임워크
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- OpenAI Simple Evals: https://github.com/openai/simple-evals
- NVIDIA NeMo Evaluator: https://github.com/NVIDIA-NeMo/Evaluator
