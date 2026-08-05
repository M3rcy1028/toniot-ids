# XAI Improvement Plan — 2026-08-05

## 목표

- 모델 학습에 사용된 22개 전체 피처와 train 147,729개 전체 행으로 중요도를 산출한다.
- SAGE를 주 선택 기준으로 사용한다.
- Class-wise SHAP과 conditional permutation feature importance(CPF)의 교집합을 클래스 보호 근거로 사용한다.
- SHAP interaction은 이미 선택된 피처의 강한 interaction partner가 누락되지 않도록 하는 보조 보호 규칙으로 사용한다.
- 1차 축소에서는 22개 피처를 18~20개로 완만하게 줄인다.

## 데이터 사용 원칙

- 중요도 산출: train 전체만 사용
- Early stopping: validation 사용
- 최종 성능 평가: 선택 규칙과 모델 설정을 확정한 뒤 test를 한 번만 사용
- SHAP과 SHAP interaction은 전체 train을 batch 처리하고 중요도 합계와 샘플 수만 누적한다.
- SAGE는 모든 클래스의 train 행이 참여하도록 클래스별로 계산하고, 각 클래스 결과를 동일 비중으로 평균한다.

## 중요도 산출 방법

### Macro/Class-wise SAGE

- 각 클래스에 대해 one-vs-rest 관점의 SAGE를 계산한다.
- 클래스별 SAGE를 동일 비중으로 평균하여 Macro-SAGE를 만든다.
- 데이터 수가 많은 normal 클래스가 결과를 지배하지 않도록 모든 클래스에 같은 가중치를 부여한다.
- 모든 22개 피처의 global/macro 및 class-wise 결과를 저장한다.

### Class-wise SHAP

- train 전체 행에 대한 multiclass SHAP 값을 batch로 계산한다.
- 각 true class에 해당하는 행과 해당 class output을 사용해 클래스별 평균 절대 SHAP 값을 계산한다.
- 모든 클래스에 동일 비중을 둔 macro SHAP과 전체 global SHAP을 함께 저장한다.

### Class-wise CPF

- train 전체 행에서 클래스별 F1 감소량을 계산한다.
- 각 피처는 상관도가 높은 conditioning feature의 quantile group 안에서만 섞는다.
- 현재 방법은 정확한 조건부분포 표본이 아니라 conditional permutation의 근사 방식임을 결과에 명시한다.

### SHAP Interaction

- train 전체 행을 batch 처리하여 클래스별 평균 절대 interaction을 집계한다.
- 유효한 multiclass output별 interaction tensor가 제공될 때만 사용한다.
- 단일 tensor를 여러 클래스에 복제하지 않는다.
- 클래스별 상위 3개 interaction pair를 보호 후보로 사용한다.

### LIME

- 1차 축소의 자동 선택에는 반영하지 않는다.
- 필요 시 오분류 원인과 개별 샘플 설명 보고서로만 사용한다.

## 1차 피처 선택 규칙

1. Macro-SAGE 상위 14개를 core feature로 선택한다.
2. 클래스별 `SHAP top12 ∩ positive CPF top12` 피처를 보호 후보로 만든다.
3. 각 클래스에 대해 보호 후보가 최소 3개 포함되도록 class-balanced consensus 점수순으로 추가한다.
4. 최종 피처가 18개보다 적으면 Macro-SAGE 순서로 18개까지 보충한다.
5. 선택된 피처가 포함된 클래스별 상위 3개 SHAP interaction pair의 상대 피처를 중요도순으로 추가한다.
6. interaction 보호를 포함한 최종 피처는 최대 20개로 제한한다.

## 실험 통제 조건

- Baseline과 동일한 train/validation/test 분할을 사용한다.
- `class_weight=None`을 유지한다.
- LightGBM hyperparameter와 early stopping 설정을 유지한다.
- 이번 실험에서는 feature selection 외의 조건을 변경하지 않는다.

## 채택 기준

- 최종 피처 수: 18~20개
- Validation accuracy 하락: 0.1%p 이하
- Validation macro F1 하락: 0.2%p 이하
- Validation MITM recall 하락: 1%p 이하
- 위 조건을 만족한 경우에만 test 결과를 최종 비교 지표로 사용한다.

## 결과 저장 위치

- XAI 결과: `reports/lightgbm_260805/xai/reduction_1`
- 축소 모델 결과: `reports/lightgbm_260805/reduction_1`
- 축소 모델: `models/lightgbm_toniot_classification_reduction_1.pkl`
- 전체 실행 로그: `260805.txt`
- 실험 기록: `ExpSheet_260805.md`

## 환경 선행 조건

- 모든 명령은 `conda activate toniot` 이후 실행한다.
- 현재 `toniot` 환경에는 Python과 패키지가 없으므로 실제 독립 환경으로 먼저 구성한다.
- SAGE 구현을 위해 `sage-importance` 의존성을 설치하고 import 및 소규모 계산을 검증한다.
- CUDA LightGBM은 현재 드라이버/library 불일치와 CUDA 미지원 빌드 때문에 사용하지 않고 CPU 8-thread로 실행한다.

## Reduction-3 계획 — 2026-08-05 17:38 KST

- 입력: validation 기준을 통과한 reduction-2의 16개 피처
- 목표: 한 단계에서 1~2개만 추가 축소하여 14~15개 피처를 선택
- 선택 규칙: Macro-SAGE top 10을 core로 하고, class-wise SHAP top 8과 positive CPF top 8의 교집합으로 클래스별 최소 3개 피처를 보호
- interaction: 유효한 multiclass SHAP interaction tensor가 있을 때만 상위 3개 pair의 partner를 보호하며, 현재 구현처럼 단일 tensor 반환 시 안전하게 생략
- 채택 기준: reduction-2 대비 validation accuracy 하락 0.1%p 이하, macro F1 하락 0.2%p 이하, MITM recall 하락 1%p 이하
- 채택된 경우에만 test를 평가하며, 기존 reduction-1/2 산출물은 덮어쓰지 않는다.
- XAI 결과: `reports/lightgbm_260805/xai/reduction_3`
- 모델 결과: `reports/lightgbm_260805/reduction_3`
- 모델: `models/lightgbm_toniot_classification_reduction_3.pkl`
- 실행 로그: `260805_reduction_3.txt`

### Reduction-3 실행 결과 — 2026-08-05 17:55 KST

- 선택: 16개 중 14개 (`dns_rcode`, `dns_RD` 제거)
- validation accuracy: 98.0320% → 97.9941% (0.0379%p 하락, 통과)
- validation macro F1: 95.8986% → 95.7218% (0.1767%p 하락, 통과)
- validation MITM recall: 79.6178% → 78.3439% (1.2739%p 하락, 실패)
- 결론: reduction-3 기각. Test 평가와 reduction-3 모델 저장을 생략하고 reduction-2 16-feature 모델을 유지한다.
