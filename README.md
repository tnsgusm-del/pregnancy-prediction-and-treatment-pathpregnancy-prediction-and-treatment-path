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

## 실행 환경 (Requirements)

```
python == 3.10.x
```

```bash
pip install -r requirements.txt
```

필요 패키지: `lightgbm`, `catboost`, `xgboost`, `torch`, `pytabkit`, `scikit-learn`, `pandas`, `numpy`

---

## 재현성 (Reproducibility)

- **트리 계열(LGBM/CatBoost/XGBoost) + Linear**: 완전 결정적이라 동일 환경에서 같은 결과가 그대로 재현됨
- **NN, TabM**: GPU/MPS 커널 특성상 완전한 결정론적 재현은 어려움. 결정성 옵션을 최대로 적용하고, `cuda → mps → cpu` 순서로 디바이스를 자동 감지하도록 구성
- **랭크 블렌딩을 쓴 이유**: 확률값은 하드웨어 차이로 미세하게 흔들릴 수 있지만 순위는 거의 안 변함. 그래서 확률이 아니라 순위를 블렌딩해서, 실행 환경이 달라져도 최종 AUC가 크게 흔들리지 않도록 설계함

---

## 최종 모델 — 6-멤버 랭크 블렌드

단일 모델이 아니라, 성격이 다른 6개 모델의 예측을 **순위(rank) 기준으로 블렌딩**했습니다.

| 순서 | 노트북 | 하는 일 | 산출물 |
|---|---|---|---|
| 1 | `1_cat_v2v3_.ipynb` | train.csv로 5개 멤버 5-Fold 학습 | `oof_*.csv`, `test_*.csv` × 5 |
| 2 | `2_TabM_전처리_통일.ipynb` | TabM 5-Fold 학습 | `oof_tabm.csv`, `test_tabm.csv` |
| 3 | `3_Hill_Climbing_FIXED.ipynb` | 6-멤버 OOF/test 로드 → 랭크 힐클라이밍 블렌드 | `submission_final.csv` |

노트북 1·2 실행 후 생성되는 `oof_*` / `test_*` 6쌍을 `data/processed/`에 모아두면, 노트북 3이 자동으로 탐색해 최종 제출 파일을 만듭니다.

### 실행 방법
```bash
# 1. 5개 멤버 학습 (Kaggle GPU 권장)
jupyter notebook 1_cat_v2v3_.ipynb

# 2. TabM 학습 (GPU/MPS/CPU 자동 감지)
jupyter notebook 2_TabM_전처리_통일.ipynb

# 3. 최종 블렌드 (CPU)
jupyter notebook 3_Hill_Climbing_FIXED.ipynb
```

---

## 기술 스택

`Python` `LightGBM` `CatBoost` `XGBoost` `PyTorch` `TabM (pytabkit)` `scikit-learn` `pandas` `NumPy`

---

## 관련 문서
- [제출 코드 상세 README (재현성·규정 준수 검증용)](./README_제출코드.md)
