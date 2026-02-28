#!/usr/bin/env python3
"""
EXAONE-4.0-1.2B 모델 구조 심층 분석

목표: 점수 향상을 위한 최적화 포인트 식별
- 레이어별 파라미터 분포
- 컴포넌트별 메모리 점유율
- 양자화 민감도 분석
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
import json

# 설정
MODEL_PATH = "./open/base_model"

def analyze_model_structure():
    """모델 구조 분석"""
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("=" * 70)
    print("EXAONE-4.0-1.2B 모델 구조 분석")
    print("=" * 70)

    # 기본 정보
    print("\n[1] 기본 구조")
    print("-" * 50)
    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Intermediate size:  {config.intermediate_size}")
    print(f"  Num layers:         {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Num KV heads:       {config.num_key_value_heads}")
    print(f"  Head dim:           {config.head_dim}")
    print(f"  Vocab size:         {config.vocab_size}")
    print(f"  GQA ratio:          {config.num_attention_heads // config.num_key_value_heads}:1")

    return config

def calculate_parameters(config):
    """레이어별 파라미터 계산"""
    h = config.hidden_size          # 2048
    ffn = config.intermediate_size  # 4096
    n_heads = config.num_attention_heads    # 32
    n_kv_heads = config.num_key_value_heads # 8
    head_dim = config.head_dim      # 64
    vocab = config.vocab_size       # 102400
    n_layers = config.num_hidden_layers  # 30

    # 레이어당 파라미터
    layer_params = {
        # Attention
        "q_proj": h * (n_heads * head_dim),      # 2048 * 2048 = 4.19M
        "k_proj": h * (n_kv_heads * head_dim),   # 2048 * 512 = 1.05M
        "v_proj": h * (n_kv_heads * head_dim),   # 2048 * 512 = 1.05M
        "o_proj": (n_heads * head_dim) * h,      # 2048 * 2048 = 4.19M

        # FFN (SwiGLU)
        "gate_proj": h * ffn,    # 2048 * 4096 = 8.39M
        "up_proj": h * ffn,      # 2048 * 4096 = 8.39M
        "down_proj": ffn * h,    # 4096 * 2048 = 8.39M

        # LayerNorm (작은 파라미터)
        "input_layernorm": h,
        "post_attention_layernorm": h,
    }

    # Embedding
    embedding_params = {
        "embed_tokens": vocab * h,  # 102400 * 2048 = 209.7M
        "final_norm": h,
    }

    # lm_head는 tie_word_embeddings=true이므로 embed_tokens와 공유

    print("\n[2] 레이어별 파라미터 분포")
    print("-" * 50)

    # Attention 총합
    attn_total = layer_params["q_proj"] + layer_params["k_proj"] + \
                 layer_params["v_proj"] + layer_params["o_proj"]

    # FFN 총합
    ffn_total = layer_params["gate_proj"] + layer_params["up_proj"] + \
                layer_params["down_proj"]

    # 레이어당 총합
    layer_total = attn_total + ffn_total + layer_params["input_layernorm"] + \
                  layer_params["post_attention_layernorm"]

    print(f"\n  [레이어당 구성요소]")
    print(f"  ├─ Attention 총합:    {attn_total:>12,} ({attn_total/1e6:.2f}M)")
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        print(f"  │  └─ {name}:        {layer_params[name]:>12,}")

    print(f"  ├─ FFN 총합:          {ffn_total:>12,} ({ffn_total/1e6:.2f}M)")
    for name in ["gate_proj", "up_proj", "down_proj"]:
        print(f"  │  └─ {name}:      {layer_params[name]:>12,}")

    print(f"  └─ 레이어 총합:       {layer_total:>12,} ({layer_total/1e6:.2f}M)")

    # 전체 모델
    all_layers_total = layer_total * n_layers
    embed_total = embedding_params["embed_tokens"] + embedding_params["final_norm"]
    model_total = all_layers_total + embed_total

    print(f"\n  [전체 모델 구성]")
    print(f"  ├─ Embedding:         {embed_total:>12,} ({embed_total/1e6:.2f}M)")
    print(f"  ├─ 30개 레이어:       {all_layers_total:>12,} ({all_layers_total/1e6:.2f}M)")
    print(f"  └─ 총 파라미터:       {model_total:>12,} ({model_total/1e9:.3f}B)")

    return layer_params, embedding_params, attn_total, ffn_total

def analyze_quantization_impact(config, layer_params, embedding_params, attn_total, ffn_total):
    """양자화 영향도 분석"""
    n_layers = config.num_hidden_layers

    print("\n[3] 양자화 영향도 분석")
    print("-" * 50)

    # 현재 설정: embed_tokens, lm_head는 FP16, 나머지는 W4
    fp16_bytes = 2
    w4_bytes = 0.5  # 4bit

    # 메모리 계산
    embed_mem = embedding_params["embed_tokens"] * fp16_bytes / 1e9

    attn_mem_fp16 = attn_total * n_layers * fp16_bytes / 1e9
    attn_mem_w4 = attn_total * n_layers * w4_bytes / 1e9

    ffn_mem_fp16 = ffn_total * n_layers * fp16_bytes / 1e9
    ffn_mem_w4 = ffn_total * n_layers * w4_bytes / 1e9

    print(f"\n  [메모리 점유율 비교]")
    print(f"  {'컴포넌트':<20} {'FP16(GB)':<12} {'W4(GB)':<12} {'점유율(W4)':<12}")
    print(f"  {'-'*56}")

    total_w4 = embed_mem + attn_mem_w4 + ffn_mem_w4

    print(f"  {'Embedding (FP16)':<20} {embed_mem:<12.3f} {embed_mem:<12.3f} {embed_mem/total_w4*100:<12.1f}%")
    print(f"  {'Attention (30층)':<20} {attn_mem_fp16:<12.3f} {attn_mem_w4:<12.3f} {attn_mem_w4/total_w4*100:<12.1f}%")
    print(f"  {'FFN (30층)':<20} {ffn_mem_fp16:<12.3f} {ffn_mem_w4:<12.3f} {ffn_mem_w4/total_w4*100:<12.1f}%")
    print(f"  {'-'*56}")
    print(f"  {'총합':<20} {'-':<12} {total_w4:<12.3f}")

    return embed_mem, attn_mem_w4, ffn_mem_w4

def suggest_optimizations(config):
    """최적화 제안"""
    print("\n[4] 점수 향상 전략")
    print("-" * 50)

    strategies = [
        {
            "name": "전략 1: 레이어별 차등 양자화 (Mixed-Precision)",
            "description": """
  - 초기 2-3개 레이어: W8A16 또는 FP16 (입력 표현에 중요)
  - 중간 레이어: W4A16 (현재와 동일)
  - 마지막 2-3개 레이어: W8A16 (출력 품질에 중요)
  - 예상 효과: PerfNorm +2~5%, SpeedNorm -1~2%
            """,
            "impact": "PerfNorm ↑↑, SpeedNorm ↓"
        },
        {
            "name": "전략 2: Attention vs FFN 차등 양자화",
            "description": """
  - Q, K projection: W8A16 (어텐션 패턴에 민감)
  - V, O projection: W4A16
  - FFN (gate, up, down): W4A16 (양자화에 robust)
  - 예상 효과: 정확도 유지하면서 속도 최적화
            """,
            "impact": "PerfNorm ↑, SpeedNorm ="
        },
        {
            "name": "전략 3: 캘리브레이션 품질 향상",
            "description": """
  - 샘플 수: 256 → 512 또는 1024
  - 시퀀스 길이: 512 → 1024 또는 2048
  - 데이터셋: MANTA-1M의 다양한 도메인 활용
  - 예상 효과: 양자화 품질 개선
            """,
            "impact": "PerfNorm ↑, SpeedNorm ="
        },
        {
            "name": "전략 4: AWQ 시도 (Marlin-AWQ)",
            "description": """
  - vLLM 벤치마크: AWQ 741 tok/s > GPTQ 712 tok/s
  - Activation-aware 양자화로 정확도 유지
  - Marlin 커널로 속도 최적화
            """,
            "impact": "SpeedNorm ↑, PerfNorm ≈"
        },
        {
            "name": "전략 5: 2:4 Structured Sparsity + Quantization",
            "description": """
  - FFN 레이어에 2:4 sparsity 적용
  - vLLM의 sparse kernel 활용
  - 양자화와 병행하여 추가 압축
  - 주의: vLLM 지원 여부 확인 필요
            """,
            "impact": "SpeedNorm ↑↑, PerfNorm ↓"
        },
    ]

    for i, s in enumerate(strategies):
        print(f"\n  [{s['name']}]")
        print(f"  영향: {s['impact']}")
        print(s['description'])

    print("\n" + "=" * 70)
    print("권장 순서: 전략 3 → 전략 1 → 전략 2 → 전략 4")
    print("=" * 70)

def analyze_layer_sensitivity():
    """레이어별 민감도 가이드"""
    print("\n[5] 레이어별 양자화 민감도 (일반적 패턴)")
    print("-" * 50)

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 0-2   : ★★★★★ 높은 민감도 (입력 표현 형성)           │
  │  Layer 3-5   : ★★★★☆ 중상 민감도                           │
  │  Layer 6-24  : ★★☆☆☆ 낮은 민감도 (안전하게 W4 적용)         │
  │  Layer 25-27 : ★★★★☆ 중상 민감도                           │
  │  Layer 28-29 : ★★★★★ 높은 민감도 (출력 품질 결정)           │
  └─────────────────────────────────────────────────────────────┘

  [권장 설정]
  - 민감 레이어 (0-2, 28-29): W8A16 또는 FP16
  - 중간 레이어 (3-27): W4A16 (현재와 동일)

  [컴포넌트별 민감도]
  - Q, K projection: ★★★★☆ (어텐션 패턴 결정)
  - V, O projection: ★★★☆☆
  - gate_proj:       ★★★☆☆ (게이팅 결정)
  - up_proj:         ★★☆☆☆
  - down_proj:       ★★☆☆☆
    """)

def main():
    config = analyze_model_structure()
    layer_params, embedding_params, attn_total, ffn_total = calculate_parameters(config)
    analyze_quantization_impact(config, layer_params, embedding_params, attn_total, ffn_total)
    analyze_layer_sensitivity()
    suggest_optimizations(config)

    print("\n\n" + "=" * 70)
    print("다음 단계: Mixed-Precision 양자화 스크립트 작성")
    print("=" * 70)

if __name__ == "__main__":
    main()
