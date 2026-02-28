# LG Aimers 8기 - LLM 경량화 해커톤

> EXAONE-4.0-1.2B 모델을 경량화하여 성능은 유지하면서 추론 속도를 극대화하는 해커톤 프로젝트

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

## 결과

**600팀 중 134등** | 최종 점수 **0.61208** (1,538명 참가)

<p align="center">
  <img src="assets/leaderboard.png" alt="DACON 리더보드" width="700">
</p>

## 프로젝트 개요

**LG Aimers 8기**에 참가하여, Phase 1(1개월 온라인 교육)과 Phase 2(온라인 해커톤)를 진행했습니다.

온라인 교육 기간 동안 학습한 AI/LLM 기초부터 경량화 기법까지의 내용은 [블로그](https://hpyquokka.tistory.com/category/LG%20Aimers)에 정리했습니다.

Phase 2 해커톤에서는 LG AI연구원의 **EXAONE-4.0-1.2B** 모델을 대상으로, GPTQ/AWQ 양자화 기법을 적용하여 **성능 유지(PerfNorm)와 속도 향상(SpeedNorm)의 최적 균형점**을 찾는 17개의 실험을 수행했습니다.

## 대회 정보

| 항목 | 내용 |
|------|------|
| 대회명 | Aimers 8기: 모델 경량화 온라인 해커톤 |
| 기간 | 2026.02.02 ~ 2026.02.26 |
| 대상 모델 | EXAONE-4.0-1.2B (1.28B params, 30 layers) |
| 평가 산식 | `Score = 0.5 × PerfNorm + 0.5 × SpeedNorm` |
| 실행 환경 | NVIDIA L4 GPU (22.4GB VRAM), 6 vCPU, 28GB RAM |
| 제약 사항 | 제출 파일 ≤ 10GB, 추론 ≤ 20분, vLLM 수정 불가 |
| 주최 / 주관 | LG AI연구원 / DACON |

## 접근 방법

17개 실험을 3단계로 나누어 체계적으로 진행했습니다.

### Phase 1: 기법 탐색 (Exp 01-04)

다양한 양자화 기법의 가능성을 탐색하는 단계입니다.

| 실험 | 기법 | 핵심 시도 | 결과 / 교훈 |
|------|------|----------|-------------|
| 01 | AWQ (활성화 인식) | actorder=static으로 AWQ 모방 | 베이스라인과 유사한 성능 |
| 02 | GPTQ group_size=64 | 더 세밀한 그룹 양자화 | Marlin 커널 비호환 발견 |
| 03 | Sparsity + Quantization | 2:4 sparse + W4A16 이중 압축 | vLLM sparse 커널 미지원 |
| 04 | Mixed-Precision | 민감 레이어(L0,1,28,29) FP16 유지 | 개념 검증 완료 |

### Phase 2: 최적화 (Exp 05-11)

Marlin 커널 호환과 파라미터 미세 조정에 집중한 단계입니다.

| 실험 | 핵심 전략 | 결과 |
|------|----------|------|
| 05 | Marlin 커널 호환 확보 (group_size=128) | SpeedNorm 대폭 향상 |
| **06** | **GPTQ 최적 파라미터 조합** | **최고 점수 0.5955 달성** |
| 07 | AWQ 스타일 (actorder=static) | GPTQ weight 방식이 우수 |
| 08 | lm_head 양자화 시도 | vLLM 로드 실패 (중요 교훈) |
| 09 | 캘리브레이션 강화 (512/1024) | 과적합으로 점수 하락 |
| 10 | 중간값 시도 (384/768) | 여전히 256/512가 최적 |
| 11 | dampening=0.0008 미세조정 | 미세한 차이 |

### Phase 3: 고급 기법 (Exp 12-17)

새로운 양자화 전략과 데이터 기반 분석을 시도한 단계입니다.

| 실험 | 핵심 전략 | 결과 |
|------|----------|------|
| 12 | W8A16 (8비트 양자화) | 성능 최대 보존, 속도 감소 |
| 13 | 민감 레이어 보호 (L0 + L29) | PerfNorm 향상 |
| 14 | 캘리브레이션 극대화 (1024/2048) | 과적합 경향 재확인 |
| 15 | 레시피 파라미터 튜닝 | 최적 조합 탐색 |
| 16 | 레이어별 민감도 분석 | 데이터 기반 보호 레이어 선정 |
| 17 | FP8 양자화 | 차세대 기법 탐색 |

## 핵심 발견

### 1. Marlin 커널 호환이 SpeedNorm의 핵심

`group_size=128`이 Marlin 커널의 필수 조건이며, Marlin 적용만으로 추론 속도가 **2.6배** 향상됩니다.

| 방식 | 처리량 | TTFT | ITL |
|------|--------|------|-----|
| Baseline FP16 | 461 tok/s | 151ms | 21.2ms |
| GPTQ (non-Marlin) | 276 tok/s | 165ms | 35.5ms |
| **Marlin-GPTQ** | **712 tok/s** | **118ms** | **13.8ms** |

### 2. 캘리브레이션 과적합의 역설

| 설정 | 결과 |
|------|------|
| 256 samples / 512 길이 | **0.5955 (최고)** |
| 384 samples / 768 길이 | 점수 하락 |
| 512 samples / 1024 길이 | 점수 하락 |

더 많은 캘리브레이션 데이터가 항상 좋은 것이 아닙니다. "적당한" 캘리브레이션이 최적입니다.

### 3. vLLM 호환성은 타협 불가

`lm_head` 양자화 시 `tie_word_embeddings`가 깨져 vLLM 로드 자체가 실패합니다.
`embed_tokens`와 `lm_head`는 반드시 양자화에서 제외해야 합니다.

### 4. 최적 설정 조합

```python
GPTQModifier(
    scheme="W4A16",
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head"],
    block_size=128,          # Marlin 호환 필수
    dampening_frac=0.001,    # 최적값
    actorder="weight",       # 정확도 향상
)
# 캘리브레이션: 256 samples, 512 seq_len
# 최고 점수: 0.5955 (추론 시간 9분 58초)
```

## 전체 버전 비교

| 버전 | actorder | dampening | group_size | samples | seq_len | 모델 크기 | 비고 |
|------|----------|-----------|------------|---------|---------|----------|------|
| Baseline | weight | 0.001 | 128 | 256 | 512 | 1.39GB | 대회 제공 |
| V1 AWQ | static | 0.01 | 128 | 256 | 512 | 1.42GB | |
| V2 GPTQ개선 | static | 0.01 | **64** | 256 | 512 | 1.42GB | Marlin 비호환 |
| V3 Sparse | - | - | 128 | 128 | 512 | 1.42GB | |
| V4 Mixed | - | - | 128 | 256 | 512 | ~1.7GB | 민감 레이어 보호 |
| V5 Marlin | weight | 0.01 | 128 | 256 | 512 | 1.42GB | |
| **V6 OptGPTQ** | **weight** | **0.001** | **128** | **256** | **512** | **1.42GB** | **최고 0.5955** |
| V7 OptAWQ | static | 0.01 | 128 | 256 | 512 | 1.42GB | |
| V8 Aggressive | weight | 0.001 | 128 | 512 | 1024 | - | lm_head 양자화 실패 |
| V9 Safe | weight | 0.001 | 128 | 512 | 1024 | 0.98GB | |
| V10 Target0.6 | weight | 0.001 | 128 | 384 | 768 | 1.42GB | |
| V11 Minimal | weight | 0.0008 | 128 | 256 | 512 | 1.42GB | |

## 기술 스택

| 구분 | 기술 |
|------|------|
| 모델 | [EXAONE-4.0-1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) |
| 양자화 | [LLM Compressor](https://github.com/vllm-project/llm-compressor) (GPTQ, AWQ) |
| 추론 엔진 | [vLLM](https://docs.vllm.ai/) + Marlin 커널 |
| 평가 | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| 캘리브레이션 데이터 | [MANTA-1M](https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M) |
| 프레임워크 | PyTorch, Transformers 4.57 |

## 프로젝트 구조

```
├── notebooks/
│   ├── 00_baseline/              # 베이스라인 GPTQ 양자화
│   ├── 01_exploration/           # 기법 탐색 (AWQ, Sparsity, Mixed-Precision)
│   ├── 02_optimization/          # 파라미터 최적화 (Marlin, 캘리브레이션 튜닝)
│   ├── 03_advanced/              # 고급 기법 (W8A16, 민감도 분석, FP8)
│   └── benchmark/                # 로컬 벤치마크
│
├── docs/                         # 대회 정보, 모델 분석, 실험 보고서
├── analyze_model.py              # 모델 구조 심층 분석 스크립트
├── setup_local.sh                # 환경 설정 스크립트
└── requirements.txt              # Python 패키지 의존성
```

## 실행 방법

```bash
# 레포 클론
git clone https://github.com/Happ11quokka/lg-aimers8-llm-compression.git
cd lg-aimers8-llm-compression

# 환경 설정
chmod +x setup_local.sh
./setup_local.sh

# 노트북 실행
source venv/bin/activate
jupyter notebook

# 모델 구조 분석
python analyze_model.py
```

> **참고**: 원본 모델(EXAONE-4.0-1.2B)은 [HuggingFace](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)에서 직접 다운로드해야 합니다. 양자화 실행에는 GPU 환경을 권장합니다.

## 블로그: LG Aimers 온라인 교육

해커톤에 앞서 1개월간 진행된 온라인 교육(Phase 1)에서 학습한 내용을 블로그에 정리했습니다.

| 날짜 | 주제 |
|------|------|
| 2026.01.18 | [AI의 첫걸음, 머신러닝과 딥러닝의 기초](https://hpyquokka.tistory.com/entry/LG-Aimers-AI%EC%9D%98-%EC%B2%AB%EA%B1%B8%EC%9D%8C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EA%B3%BC-%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%98-%EA%B8%B0%EC%B4%88) |
| 2026.01.19 | [Decoding of Large Language Models](https://hpyquokka.tistory.com/entry/LG-Aimers-Decoding-of-Large-Language-Models) |
| 2026.01.22 | [경량화 LLM/SLM, "작게 잘 쓰는" 게 전략이다](https://hpyquokka.tistory.com/entry/%EA%B2%BD%EB%9F%89%ED%99%94-LLMSLM-%EC%9D%B4%EC%A0%9C-%E2%80%9C%EC%9E%91%EA%B2%8C-%EC%9E%98-%EC%93%B0%EB%8A%94%E2%80%9D-%EA%B2%8C-%EC%A0%84%EB%9E%B5%EC%9D%B4%EB%8B%A4) |
| 2026.01.27 | [Lightweight LLM: 스케일링 이후의 승부처](https://hpyquokka.tistory.com/entry/Lightweight-LLM-%E2%80%9C%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81-%EC%9D%B4%ED%9B%84%E2%80%9D%EC%9D%98-%EC%8A%B9%EB%B6%80%EC%B2%98%EB%8A%94-%E2%80%98%EC%9E%91%EA%B2%8C-%EB%B9%A0%EB%A5%B4%EA%B2%8C-%EC%8B%B8%EA%B2%8C%E2%80%99%EC%98%80%EB%8B%A4) |
| 2026.01.27 | [Breaking Scaling Law: Distillation으로 가는 길](https://hpyquokka.tistory.com/entry/Breaking-Scaling-Law-%E2%80%9C%ED%81%AC%EA%B2%8C%E2%80%9D%EC%97%90%EC%84%9C-%E2%80%9C%EB%98%91%EB%98%91%ED%95%98%EA%B2%8C%EC%8B%B8%EA%B2%8C%E2%80%9D%EB%A1%9C-%E2%80%94-Distillation%EB%A1%9C-%EA%B0%80%EB%8A%94-%EA%B8%B8) |
| 2026.01.30 | [초거대 언어 모델(LLM) 압축 기법](https://hpyquokka.tistory.com/entry/CLLM) |

## 참고 자료

### 논문
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Frantar & Alistarh, "SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot" (2023)

### 문서
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor)
- [DACON 대회 페이지](https://dacon.io/competitions/official/236473/overview/description)

---

*이 프로젝트는 LG Aimers 8기 온라인 해커톤(Phase 2)에 참가하며 진행한 개인 실험 기록입니다.*
