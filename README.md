# Cross-Lingual Transfer Experiments: Zero-Shot to Full Fine-Tuning
> XLM-RoBERTa 기반 다국어 NER에서 교차 언어 전이학습 성능 비교

이 프로젝트는 XLM-RoBERTa(XLM-R) 모델을 활용해 다국어 개체명 인식(NER) 태스크에서 언어 간 전이학습(Cross-Lingual Transfer Learning)의 성능을 실험적으로 비교한 연구입니다.  
Zero-shot, Few-shot, Multilingual, Full fine-tuning 전략을 동일한 파이프라인에서 정량적으로 분석했습니다.

---

## 프로젝트 개요
교차 언어 전이학습은 한 언어에서 학습한 모델의 지식을 다른 언어로 확장하는 기술입니다.  
본 연구에서는 저자원 언어 환경에서의 효율적인 NER 학습을 목표로, 언어별 학습 데이터 비율을 조정하고 전이 전략의 효율성을 비교했습니다.

---

## 실험 설계
| 항목 | 설정 |
|---|---|
| **모델** | xlm-roberta-base |
| **태스크** | Named Entity Recognition (BIO 스키마: B/I-PER, B/I-ORG, B/I-LOC, O) |
| **데이터셋** | PAN-X (WikiAnn) |
| **언어** | 독일어(de), 프랑스어(fr), 이탈리아어(it), 영어(en) |
| **평가지표** | F1-score (seqeval 라이브러리) |

의도적으로 데이터 불균형을 구성하여 실제 시나리오를 모사했습니다.  
비율은 독일어 62.9%, 프랑스어 22.9%, 이탈리아어 8.4%, 영어 5.9%로 설정했습니다.

---

## 훈련 파이프라인
1. **데이터 전처리 및 정렬**  
   - XLM-R 서브워드 토크나이저 사용  
   - 첫 서브워드에만 레이블 부여, 나머지는 `-100`으로 마스킹 처리  
2. **데이터 인코딩**  
   - Hugging Face `datasets.map(batched=True)`를 사용한 병렬 인코딩  
3. **모델 정의**  
   - `XLMRobertaForTokenClassification` 커스텀 클래스 구현  
   - Dropout + Linear head 구성  
4. **학습 구조**  
   - `Trainer` API와 `TrainingArguments`, `DataCollatorForTokenClassification` 활용  
5. **평가**  
   - `seqeval` 기반 F1-score 계산  

---

## 실험 결과

### Zero-Shot (de로 학습 → 타 언어 평가)
| 평가 언어 | de | fr | it | en |
|---|---:|---:|---:|---:|
| F1-score | **0.875** | 0.704 | ~0.70 | 0.606 |

---

### Few-Shot (fr 샘플 수 증가)
| 프랑스어 샘플 수 | 250 | 500 | 750 | 1,000 | 2,000 | 4,000 |
|---:|---:|---:|---:|---:|---:|---:|
| F1-score | ~0.65 | ~0.68 | **~0.70** | ~0.72 | ~0.76 | ~0.80 |

---

### Multilingual (de+fr 동시 학습)
| 평가 언어 | de | fr | it | en |
|---|---:|---:|---:|---:|
| F1-score | **0.876** | **0.865** | 0.784 | 0.661 |

> Full multilingual(de+fr+it+en) 실험은 환경에 따라 변동 가능하지만, 전반적으로 모든 언어에서 균형 잡힌 성능을 보입니다.

---

## 분석 및 인사이트
1. **Zero-shot의 효율성**  
   - 레이블링 없이도 중간 수준의 F1 달성  
   - 초기 배포 단계에 유리  

2. **Few-shot의 임계점**  
   - 약 750개의 샘플에서 Zero-shot과 유사한 성능  
   - 1,000개 이상부터 점진적 개선이 있으나 체감 효율은 감소  

3. **Multilingual의 시너지 효과**  
   - 언어 간 지식 공유로 모든 언어의 성능 향상  
   - 미관측 언어(it, en)에서도 일반화 이득 발생  

4. **성능 차이 요인**  
   - 언어적 거리(게르만어 ↔ 로망어)  
   - 라벨 분포와 토크나이저 분절 차이  
   - 형태소 및 표기 규칙의 차이  

5. **권장 전략 요약**  
   - **예산 제한:** Zero-shot 또는 소량 Few-shot  
   - **균형 성능:** Multilingual Fine-tuning  
   - **도메인 특화:** Few-shot으로 보정  

---

## 기술 스택
| 구성 | 내용 |
|---|---|
| **Framework** | PyTorch, Transformers, Datasets |
| **Evaluation** | seqeval |
| **Utilities** | pandas, numpy |
| **환경** | Python 3.8+, CUDA GPU, Colab 또는 로컬 실행 가능 |

---



