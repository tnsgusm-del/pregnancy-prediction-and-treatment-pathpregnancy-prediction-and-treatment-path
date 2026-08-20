# 난임치료 성공 예측 모델 (Pregnancy Success Forecaster)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-gray)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-yellow)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-green)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TabM%20%7C%20NN-orange)](https://pytorch.org/)
[![Result](https://img.shields.io/badge/Public%20LB-0.7425-brightgreen)]()

> 난임(불임) 환자의 임상·시술 데이터를 기반으로 **임신 성공 여부를 예측**하는 이진 분류 모델입니다.
> 해커톤 참가 결과 **전체 2위 (Public LB AUC 0.7425)**를 기록했습니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **과제** | 난임 시술 데이터를 통한 임신 성공/실패 이진 분류 |
| **평가지표** | ROC-AUC |
| **최종 성적** | **2위** (Public LB **0.7424857289**) |
| **담당 역할** | 피처 엔지니어링, 멀티 모델 학습 파이프라인 설계, 앙상블 가중치 최적화 |

단순 수치 처리를 넘어 도메인 맥락이 반영된 피처를 설계하고, 이를 6개 모델의 이종 앙상블로 통합한 프로젝트입니다.

---

## 최종 모델 — 6-멤버 랭크 블렌드

단일 모델이 아니라, 성격이 다른 6개 모델의 예측을 **순위(rank) 기준으로 블렌딩**했습니다.

```
LightGBM · CatBoost · XGBoost · Linear(ratio) · NN(임베딩-MLP) · TabM
```

가중치는 OOF(Out-of-Fold) 예측에 대해 **Hill Climbing 탐색**으로 직접 최적화했습니다.

| 멤버 | LightGBM | CatBoost | XGBoost | Linear | NN | **TabM** | **6-멤버 블렌드** |
|---|---|---|---|---|---|---|---|
| OOF AUC | 0.73965 | 0.73974 | 0.73964 | 0.72029 | 0.73812 | **0.74002** | **0.74090** |
| 블렌드 가중치 | 0.200 | 0.212 | 0.094 | 0.012 | 0.141 | **0.341** | - |

- **TabM**(tabular 전용 딥러닝, `pytabkit`)이 단일 모델 중 최고 성능이자 최대 가중치를 차지
- 트리 계열(LGBM/CatBoost/XGBoost) + 선형 + 신경망 계열을 모두 섞어 **모델 다양성으로 일반화 성능 확보**
- 최종 제출 점수: **Public LB 0.7424857289**

---

## 핵심 설계 포인트

### 1. 랭크(Rank) 블렌딩으로 딥러닝 비결정성 방어
NN·TabM은 GPU/MPS 커널 특성상 완전한 결정론적 재현이 어렵습니다. 확률값을 그대로 평균 내면 하드웨어 차이로 점수가 흔들릴 수 있어, **확률이 아닌 순위를 블렌딩**하는 방식을 채택했습니다. 개별 확률이 소수점 단위로 흔들려도 순위는 거의 불변이기 때문에, 재현 환경이 달라져도 AUC가 안정적으로 보존되도록 설계했습니다.

### 2. Fold-내부 Fit으로 데이터 누수 원천 차단
인코딩·스케일링·결측치 대치 등 모든 전처리를 **매 fold의 train 데이터에서만 학습(fit)**하고 valid/test에는 transform만 적용했습니다. 학습 직후 `AUC < 0.999` 자동 검증(assert)을 걸어, 혹시 모를 라벨 누수를 코드 레벨에서 즉시 탐지하도록 했습니다.

### 3. 학습 환경 분리 & 재현성 확보
- 트리·선형 모델: 완전 결정적 (동일 환경에서 bit-동일 재현)
- NN·TabM: 결정성 옵션 최대 적용 + `cuda → mps → cpu` 자동 디바이스 감지
- 서로 다른 학습 환경(Kaggle GPU / 로컬 CUDA·MPS)을 노트북 단위로 분리하여 관리

### 4. 30+ 실험 단계에 걸친 반복 개선
피처 엔지니어링 → 그룹별 모델 블렌딩 → 확률 캘리브레이션 → 앙상블 가중치 최적화까지, 30회 이상의 ablation을 거쳐 최종 파이프라인에 도달했습니다.

---

## 프로젝트 구조 & 실행 순서

```
├── 1_cat_v2v3_.ipynb              # [1/3] 5개 멤버(LGBM·Cat·XGB·Lin·NN) 학습 — Kaggle GPU
├── 2_TabM_전처리_통일.ipynb          # [2/3] TabM 학습 — CUDA/MPS/CPU 자동 감지
├── 3_Hill_Climbing_FIXED.ipynb    # [3/3] 6-멤버 랭크 블렌드 → 최종 제출 파일 생성
└── README.md
```

| 순서 | 노트북 | 하는 일 | 산출물 |
|---|---|---|---|
| 1 | `1_cat_v2v3_.ipynb` | train.csv로 5개 멤버 5-Fold 학습 | `oof_*.csv`, `test_*.csv` × 5 |
| 2 | `2_TabM_전처리_통일.ipynb` | TabM 5-Fold 학습 | `oof_tabm.csv`, `test_tabm.csv` |
| 3 | `3_Hill_Climbing_FIXED.ipynb` | 6-멤버 OOF/test 로드 → 랭크 힐클라이밍 블렌드 | `submission_final.csv` |

> 노트북 1·2 실행 후 생성되는 `oof_*` / `test_*` 6쌍을 같은 폴더(또는 `data/`)에 모아두면, 노트북 3이 자동으로 탐색하여 최종 제출 파일을 생성합니다.

### 실행 방법
```bash
# 1. 5개 멤버 학습 (Kaggle GPU 권장)
jupyter notebook 1_cat_v2v3_.ipynb

# 2. TabM 학습 (GPU/MPS/CPU 자동 감지)
jupyter notebook 2_TabM_전처리_통일.ipynb

# 3. 최종 블렌드 (CPU, 수초 소요)
jupyter notebook 3_Hill_Climbing_FIXED.ipynb
```

---

## 기술 스택

`Python` `LightGBM` `CatBoost` `XGBoost` `PyTorch` `TabM (pytabkit)` `scikit-learn` `pandas` `NumPy`

---

## 관련 문서
- [제출 코드 상세 README (재현성·규정 준수 검증용)](./README_제출코드.md)
