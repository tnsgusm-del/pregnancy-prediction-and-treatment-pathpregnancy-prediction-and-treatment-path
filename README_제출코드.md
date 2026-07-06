# 제출 코드 — 난임 임신성공 예측 (6멤버 랭크 블렌드)

팀 «1단 임신 성공» · 최종 리더보드 제출 **Public LB 0.7424857289** 를 재현하는 코드입니다.

이 코드 묶음의 목적은 두 가지입니다.
1. **규정 준수 확인** — 참가자가 대회 규정(test 누수 차단 등)에 맞게 코드를 작성했는지, 코드 수준에서 검증할 수 있도록 구성했습니다.
2. **점수 재현 확인** — 주최측이 제출 코드를 순서대로 실행했을 때, 제출한 리더보드 점수와 동일한 결과(`submission_final.csv`)가 재현되도록 구성했습니다.

실제 학습이 서로 다른 환경(케글 GPU / 로컬·MPS·CUDA)에서 이루어졌기 때문에, 환경 보존과 재현성 검증을 위해 **3개 파일**로 분리했습니다.

---

## 최종 모델

**6멤버 랭크(rank) 블렌드** — `lgb(v2v3) · cat(v2v3) · xgb(v3) · lin(ratio) · nn(v2v3) · tabm(전처리 통일)`

OOF(train 라벨만) 힐클라이밍으로 산출한 가중치:

```
{ lgb 0.200, cat 0.212, xgb 0.094, lin 0.012, nn 0.141, tabm 0.341 }
```

멤버별 OOF AUC (파일 3 로드 시 출력 · 5-fold seed42):

| 멤버 | lgb | cat | xgb | lin | nn(v2v3) | tabm | **6멤버 블렌드** |
|---|---|---|---|---|---|---|---|
| OOF AUC | 0.73965 | 0.73974 | 0.73964 | 0.72029 | 0.73812 | 0.74002 | **0.74090** |

> 최종 제출 점수: **Public LB 0.7424857289** (`submission_final.csv`).

---

## 실행 순서

| 순서 | 파일 | 환경 | 하는 일 | 산출물 |
|---|---|---|---|---|
| 1 | `1_cat_v2v3_.ipynb` | 케글 GPU 세션(NN) · 트리·선형 CPU | `train.csv`에서 5멤버(lgb·cat·xgb·lin·nn) 5-fold 학습 | `oof_{m}.csv`·`test_{m}.csv` ×5 |
| 2 | `2_TabM_전처리_통일.ipynb` | GPU/MPS/CPU 자동 감지 | `train.csv`에서 TabM(전처리 통일) 5-fold 학습 | `oof_tabm.csv`·`test_tabm.csv` |
| 3 | `3_Hill_Climbing_FIXED.ipynb` | CPU 어디서나 (수초) | 6멤버 OOF/test 로드 → 랭크 힐클라이밍 블렌드 | `submission_final.csv` (+ `submission_no_lin.csv`) |

> - 파일 3은 파일 1·2의 산출물(`oof_*`·`test_*` 6쌍)을 입력으로 받습니다. **같은 폴더(또는 `data/`)에 모아 두고** 파일 3을 실행하세요.
> - 입력 `train.csv`·`test.csv`는 세 파일 모두 필요합니다.
> - `find_csv()` 헬퍼가 `./` · `data/` · `../data/` · `/kaggle/input/` · `/kaggle/working/` 및 재귀 글롭으로 입력 파일을 자동 탐색하므로, 경로를 수정하지 않아도 됩니다.

---

## 환경 (실제 학습 환경 보존)

- **트리·선형·NN** = 파일 1 (케글 GPU 세션; NN은 GPU, 트리·선형은 CPU에서 학습)
- **TabM** = 파일 2 (`2_TabM_전처리_통일`; quantile·robust 전처리를 통일한 버전. 디바이스는 `cuda → mps → cpu` 순으로 자동 감지)
- **블렌드** = 파일 3 (CPU 어디서나, 수초)
- 주요 라이브러리 버전은 각 파일이 `env_versions*.json`으로 스냅샷합니다. (TabM은 `pytabkit`, Apache-2.0)

---

## 규정 준수 (test 누수 차단) — 코드로 검증 가능

- **test는 어떤 학습/전처리 fit에도 들어가지 않습니다.** 명목형 인코딩·스케일·결측 대치·타깃인코딩·TabM 내부 전처리(quantile·robust)는 전부 **매 fold의 train fold에서만 fit** 후 valid/test에 transform합니다(각 코드에 `# 규정: train-only fit` 취지의 주석).
- **외부 데이터·유사라벨링(pseudo-labeling) 미사용.**
- 멤버 학습 직후 `AUC < 0.999` assert로 라벨 누수를 자동 차단합니다.
- 5멤버는 동일한 `seed42 StratifiedKFold(5)`로 OOF 행이 정렬되며, 파일 3 로드 시 멤버 간 `y`(정답) 일치를 assert로 교차검증합니다(`oof의 y가 다른 멤버와 다릅니다!` 가드).
- `OCC/AGE` 순서맵은 데이터 비의존 고정 사전이라 누수가 없습니다.

---

## 재현성 — 무엇이 bit-동일하고 무엇이 하드웨어 의존인가

- **트리·선형(lgb·cat·xgb·lin)**: 시드·스레드수 고정으로 **완전 결정적** — 동일 환경에서 bit-동일 재현.
- **NN·TabM**: 결정성 풀세트(`use_deterministic_algorithms`·`cudnn.deterministic`·`CUBLAS_WORKSPACE_CONFIG`·`random_state`)를 적용하되, 일부 GPU/MPS 커널은 결정적 구현이 없습니다. 따라서 **GPU 종류·드라이버·CUDA/MPS 차이만으로 확률이 소수점 4째자리에서 흔들릴 수 있습니다.** (TabM은 엄격 결정성 시 크래시하는 커널이 있어, 경고 후 비결정 모드로 폴백합니다.)
- **방어 — 랭크(rank) 블렌딩**: 최종 블렌드는 확률 평균이 아니라 **순위(rank) 평균**입니다. 딥 멤버 확률이 미세히 변해도 *순위*는 거의 불변이라 **AUC(점수)가 보존**됩니다. 비결정성을 완전히 제거할 수는 없으나, 블렌드 구조로 그 영향을 흡수하도록 설계했습니다.
- 다른 하드웨어에서 재실행 시 점수가 소수점 4째자리에서 달라질 수 있으나, 랭크 블렌딩으로 **~0.7424 수준은 보존**됩니다.

---

## 점수 재현 절차 (주최측 검증용)

1. `train.csv`·`test.csv`와 세 노트북을 같은 작업 폴더에 둡니다.
2. 파일 1 → 파일 2를 실행해 멤버별 `oof_*`·`test_*` (6쌍)을 생성합니다.
3. 파일 3을 실행하면 6멤버 랭크 힐클라이밍 블렌드로 **`submission_final.csv`** 가 생성됩니다. 이 파일이 제출 점수 **Public LB 0.7424857289** 에 대응합니다.
4. 규정 준수 여부는 위 「규정 준수」 항목(test 미투입 · fold-내부 fit · `AUC<0.999` assert · 멤버 간 `y` 일치 assert)으로 코드 수준에서 확인할 수 있습니다.
5. (보너스) 파일 3은 가중치가 낮은 `lin` 제외 5멤버 블렌드도 자동 검증해 `submission_no_lin.csv`로 저장합니다. 6멤버 블렌드와 OOF가 동일(0.74090)함을 출력으로 확인할 수 있습니다.

---

## 산출 파일

- `submission_final.csv` — **최종 제출 파일** (파일 3)
- `submission_no_lin.csv` — (선택) LIN 제외 검증본, OOF 동일 (파일 3)
- `reproduce_report.json` — 가중치·멤버 OOF·환경·재현성 주석 (파일 3)
- `oof_*.csv` / `test_*.csv` — 멤버별 OOF/test 예측 = 누수안전 증빙 (파일 1·2)

---

*비고: 일부 노트북 상단 설명 셀에는 이전 버전의 문구(예: 옛 LB 0.7422428, `nn(base)`, TabM `Mac(MPS)` 표기)가 남아 있을 수 있습니다. 실제 산출물 기준으로는 본 README가 정확합니다 — `nn` 멤버는 파일 1에서 `members_oof["nn"]=oof_nn_v2v3`로 v2v3 버전이 저장되고, TabM은 전처리 통일 버전(OOF 0.74002)이며, 최종 블렌드 OOF는 0.74090, 제출 점수는 0.7424857289입니다.*
