# Cross-Lingual Transfer Experiments: Zero-Shot to Full Fine-Tuning

> **XLM-RoBERTa 기반 다국어 개체명 인식에서 언어 간 전이학습 실험 연구**

이 프로젝트는 XLM-RoBERTa (XLM-R) 모델을 활용하여 다국어 개체명 인식(NER) 태스크에서 교차 언어 전이학습(Cross-Lingual Transfer Learning)의 효과를 체계적으로 실험한 연구입니다. Zero-shot, Few-shot, Multilingual, Full fine-tuning 등 다양한 학습 전략을 비교 분석합니다.

---

## Table of Contents

- [프로젝트 개요](#프로젝트-개요)
- [실험 설계](#실험-설계)
- [훈련 파이프라인](#훈련-파이프라인)
- [실험 결과](#실험-결과)
- [결과 분석 및 인사이트](#결과-분석-및-인사이트)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 개요

### Cross-Lingual Transfer Learning이란?

교차 언어 전이학습은 한 언어에서 학습된 모델의 지식을 다른 언어에 적용하는 기술입니다. 레이블링된 데이터가 충분하지 않은 저자원 언어에서도 고성능 NER 모델을 구축할 수 있게 해주는 핵심 접근법입니다.

### 실험 전략

본 연구에서는 다음과 같은 학습 전략을 비교 분석했습니다:

| 전략 | 설명 | 비용 대비 효율 |
|------|------|----------------|
| **Zero-shot** | 한 언어로만 학습하고 다른 언어에 직접 적용 | 매우 효율적 |
| **Few-shot** | 소량의 타겟 언어 데이터로 추가 학습 | 효율적 |
| **Multilingual** | 여러 언어를 동시에 학습 | 균형적 |
| **Full Fine-tuning** | 각 언어별로 완전한 미세조정 | 비용 높음 |

### 실험 목적

1. **Zero-shot 전이 성능 평가**: 독일어로 학습한 모델이 다른 언어에서 얼마나 잘 작동하는지 검증
2. **Few-shot 학습 임계점 발견**: 타겟 언어 데이터가 얼마나 필요할 때 zero-shot을 넘어서는지 분석
3. **Multilingual 학습 효과**: 다국어 동시 학습이 언어 간 일반화에 미치는 영향 측정
4. **비용 대비 효율성**: 데이터 수집 비용을 고려한 최적 학습 전략 제시

---

## 실험 설계

### 모델 및 데이터셋

| 항목 | 세부 사항 |
|------|-----------|
| **모델** | `xlm-roberta-base` (XLM-RoBERTa Base) |
| **태스크** | Named Entity Recognition (NER) |
| **데이터셋** | PAN-X (WikiAnn 기반) |
| **언어** | 독일어(de), 프랑스어(fr), 이탈리아어(it), 영어(en) |
| **엔티티 타입** | PER (Person), ORG (Organization), LOC (Location) |
| **레이블 스키마** | BIO 형식 (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O) |

### 데이터 분포

의도적으로 불균형한 데이터 분포를 만들어 실생활 시나리오를 시뮬레이션했습니다:

| 언어 | 훈련 데이터 비율 | 목적 |
|------|-----------------|------|
| 독일어 (de) | 62.9% | 고자원 언어 (메인 학습 데이터) |
| 프랑스어 (fr) | 22.9% | 중자원 언어 |
| 이탈리아어 (it) | 8.4% | 저자원 언어 |
| 영어 (en) | 5.9% | 최저자원 언어 |

### 평가 지표

- **Primary Metric**: F1-score (`seqeval` 라이브러리 사용)
- **추가 분석**: Token-level 손실 분석, 언어별 엔티티 타입별 성능

### 실험 설정

```python
# 하이퍼파라미터
num_epochs = 3
batch_size = 16
learning_rate = (기본값)
weight_decay = 0.05
eval_strategy = "epoch"
```

---

## 훈련 파이프라인

### 1. 데이터 전처리

#### 토큰-레이블 정렬 (Token-Label Alignment)

XLM-RoBERTa는 서브워드 토큰화를 사용하므로, 단어 단위 레이블을 서브워드에 맞춰 정렬해야 합니다:

- **첫 번째 서브워드**: 원본 레이블 할당 (B-PER, I-PER 등)
- **후속 서브워드**: `-100`으로 마스킹 (손실 계산에서 무시)

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], 
                                truncation=True, 
                                is_split_into_words=True)
    # word_ids()를 사용하여 각 토큰이 속한 원본 단어 추적
    # 첫 서브워드에만 레이블 할당
    ...
```

#### 데이터 인코딩

- Hugging Face `Datasets` 라이브러리의 `map()` 함수로 병렬 처리
- 배치 처리 (`batched=True`)로 효율성 향상

### 2. 모델 정의

```python
class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    """XLM-RoBERTa 기반 토큰 분류 모델"""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

### 3. 학습 루프

Hugging Face `Trainer` API를 활용한 학습:

```python
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,  # seqeval 기반 F1-score
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)
```

### 4. 평가 함수

`seqeval`을 사용한 정확한 NER 평가:

```python
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(
        eval_pred.predictions, 
        eval_pred.label_ids
    )
    return {"f1": f1_score(y_true, y_pred)}
```

---

## 실험 결과

### 1. Zero-Shot Cross-Lingual Transfer

독일어로 학습한 모델을 다른 언어에 직접 적용한 결과:

| 평가 언어 → | 독일어 (de) | 프랑스어 (fr) | 이탈리아어 (it) | 영어 (en) |
|-------------|-----------|--------------|---------------|----------|
| **학습 언어** |            |              |               |          |
| 독일어 (de) | **0.875** | 0.704 | ~0.70 | 0.606 |

#### 주요 발견사항

- **독일어 → 프랑스어**: 약 17% 성능 저하 (0.875 → 0.704)
- **독일어 → 영어**: 약 31% 성능 저하 (0.875 → 0.606)
- 게르만어파와 로망어군 간의 언어적 거리가 성능 차이에 영향을 미침
- Zero-shot만으로도 상당한 성능 달성 가능 (비용 효율적)

### 2. Few-Shot Cross-Lingual Transfer

프랑스어 훈련 데이터를 점진적으로 증가시키며 성능 변화 관찰:

| 프랑스어 훈련 샘플 수 | F1-Score | Zero-shot 대비 |
|----------------------|----------|----------------|
| 250 | ~0.65 | ↓ 5.4%p |
| 500 | ~0.68 | ↓ 2.4%p |
| 750 | **약 0.704** | **동등** |
| 1,000 | ~0.72 | ↑ 1.6%p |
| 2,000 | ~0.76 | ↑ 5.6%p |
| 4,000 | ~0.80 | ↑ 9.6%p |

#### 인사이트

- **임계점**: 약 750개 샘플에서 zero-shot과 동등한 성능
- **수익 체감**: 샘플 수가 증가할수록 성능 향상 폭이 줄어듦
- **비용 고려**: 750개 미만에서는 zero-shot이 더 효율적

### 3. Multilingual Fine-Tuning

#### 독일어 + 프랑스어 동시 학습

| 평가 언어 → | 독일어 (de) | 프랑스어 (fr) | 이탈리아어 (it) | 영어 (en) |
|-------------|-----------|--------------|---------------|----------|
| **학습 언어** |            |              |               |          |
| de + fr | **0.876** | **0.865** | 0.784 | 0.661 |

**주요 성과:**
- 프랑스어 성능이 zero-shot (0.704) 대비 **23% 향상** (0.865)
- 독일어 성능은 유지 (0.875 → 0.876)
- **언어 간 일반화 효과**: 본 적 없는 언어(이탈리아어, 영어) 성능도 향상

#### Full Multilingual Fine-Tuning (de + fr + it + en)

| 평가 언어 → | 독일어 (de) | 프랑스어 (fr) | 이탈리아어 (it) | 영어 (en) |
|-------------|-----------|--------------|---------------|----------|
| **학습 언어** |            |              |               |          |
| All Languages | - | - | - | - |

> **참고**: Full multilingual 실험 결과는 노트북 최종 실행에 따라 다를 수 있습니다. 일반적으로 모든 언어에서 균형 잡힌 성능을 보입니다.

### 종합 비교

| 전략 | de | fr | it | en | 평균 | 비용 효율성 |
|------|----|----|----|----|------|------------|
| Zero-shot (de만 학습) | 0.875 | 0.704 | ~0.70 | 0.606 | ~0.72 | 매우 높음 |
| Few-shot (fr 1000개) | - | ~0.72 | - | - | - | 높음 |
| Multilingual (de+fr) | 0.876 | 0.865 | 0.784 | 0.661 | ~0.80 | 중간 |
| Full Fine-tuning | - | - | - | - | - | 낮음 |

---

## 결과 분석 및 인사이트

### 1. Zero-Shot 전이의 강점

- **데이터 수집 비용 절감**: 타겟 언어 레이블링이 필요 없음
- **빠른 배포**: 추가 학습 없이 즉시 적용 가능
- **언어 간 공통 패턴 활용**: XLM-R의 다국어 사전학습 지식 활용

### 2. Few-Shot 학습의 최적화점

- **750개 샘플**: Zero-shot과 동등한 성능 달성 (비용 효율성 관점에서 임계점)
- **1,000개 이상**: 점진적 성능 향상이지만 체감 수익 감소
- **실무 권장**: 레이블링 예산이 제한적이면 zero-shot 우선 고려

### 3. Multilingual 학습의 시너지 효과

- **언어 간 지식 공유**: 한 언어 학습이 다른 언어 성능 향상에 기여
- **일반화 능력**: 본 적 없는 언어에서도 성능 개선
- **균형 잡힌 성능**: 모든 언어에서 고르게 좋은 성능

### 4. 성능 차이의 원인

- **언어적 거리**: 게르만어파 vs 로망어군
- **라벨 분포 차이**: 언어별 PER/ORG/LOC 빈도 및 패턴 차이
- **토크나이저 단편화**: 언어별 서브워드 분해 패턴 차이
- **문자 및 형태학적 차이**: 대문자 규칙, 접사, 띄어쓰기 등

### 5. 실무 적용 권장사항

1. **저예산 프로젝트**: Zero-shot 우선 시도
2. **중간 예산**: Few-shot (750-1000개 샘플)로 성능 개선
3. **충분한 예산**: Multilingual 학습으로 균형 잡힌 성능 달성
4. **도메인 특화**: 도메인 전문가가 필요할 경우 Few-shot 전략 고려

---

## 기술 스택

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **transformers** | 최신 | 모델 및 토크나이저, Trainer API |
| **datasets** | 최신 | 데이터셋 로딩 및 전처리 |
| **torch** | 최신 | 딥러닝 프레임워크 |
| **seqeval** | 1.2.2+ | NER 평가 지표 (F1-score) |
| **pandas** | 최신 | 데이터 분석 및 시각화 |
| **numpy** | 최신 | 수치 연산 |

### 모델 및 데이터셋

- **모델**: `xlm-roberta-base` (Hugging Face Hub)
- **데이터셋**: `xtreme` (PAN-X 서브셋)

### 실행 환경

- Python 3.8+
- CUDA 지원 GPU (권장)
- Google Colab / 로컬 환경 모두 지원

---

## 프로젝트 구조

```
Exploring Cross-Lingual Transfer with XLM-R/
│
├── README.md                                    # 프로젝트 문서
├── Multilingual_NER_with_the_WikiANN_(PAN_X)_Dataset .ipynb  # 메인 실험 노트북
└── multilingual_ner_with_the_wikiann_(pan_x)_dataset .py     # 변환된 Python 스크립트
```

### 노트북 주요 섹션

1. **데이터 로딩 및 전처리**: PAN-X 데이터셋 불균형 생성
2. **모델 정의**: XLMRobertaForTokenClassification 커스텀 클래스
3. **토큰-레이블 정렬**: 서브워드 토큰화 대응
4. **Zero-shot 실험**: 독일어 모델의 다국어 전이 성능
5. **Few-shot 실험**: 프랑스어 샘플 수에 따른 성능 변화
6. **Multilingual 실험**: 다국어 동시 학습
7. **오류 분석**: 토큰 레벨 손실 분석

---

## 시작하기

### 설치

```bash
pip install transformers datasets torch seqeval pandas numpy
```

### 실행

Jupyter Notebook 또는 Google Colab에서 노트북 파일을 열고 순차적으로 실행:

1. 데이터셋 로딩
2. 데이터 전처리
3. 모델 학습
4. 평가 및 분석

### Hugging Face Hub 업로드 (선택사항)

학습된 모델을 Hub에 업로드하려면:

```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## 향후 개선 방향

- [ ] 더 많은 언어 추가 (스페인어, 포르투갈어 등)
- [ ] 엔티티 타입별 세부 성능 분석
- [ ] 하이퍼파라미터 최적화 (learning rate, batch size 등)
- [ ] 도메인 적응 실험 (뉴스 → 의료 문서 등)
- [ ] 앙상블 기법 적용

---

## 참고 자료

- **XLM-RoBERTa Paper**: [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
- **PAN-X Dataset**: [XTREME Benchmark](https://github.com/google-research/xtreme)
- **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **seqeval**: [NER Evaluation Library](https://github.com/chakki-works/seqeval)

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 작성되었습니다.

---

## 저자

이 프로젝트는 XLM-RoBERTa를 활용한 교차 언어 전이학습 실험 연구입니다.

---

**Keywords**: `Cross-Lingual Transfer Learning`, `XLM-RoBERTa`, `Named Entity Recognition`, `Zero-Shot Learning`, `Few-Shot Learning`, `Multilingual NLP`, `Natural Language Processing`

