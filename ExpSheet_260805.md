# Experiment Sheet — 2026-08-05

## 2026-08-05 16:11 KST — Train/Validation/Test 분할 및 LightGBM 학습 개선

### 변경 사항

- `toniot_dataset.py`의 데이터 경로를 실제 네트워크 데이터 위치인 `data_network/raw`와 `data_network/processed_type`으로 통일했다.
- 원본 211,043개 행을 클래스 비율을 유지하는 stratified 방식으로 분할했다.
  - Train: 147,729개 (70%)
  - Validation: 31,657개 (15%)
  - Test: 31,657개 (15%)
- `X_valid.csv`와 `y_valid.csv`를 새로 저장하도록 전처리 파이프라인을 확장했다.
- 범주형 인코더와 MinMaxScaler는 train 데이터에만 fit하고 validation/test에는 transform만 적용하도록 구성했다.
- `experiment.py`가 train/validation/test 여섯 개 파일을 모두 읽고, feature selection 사용 시 세 분할에 동일한 피처 목록을 적용하도록 수정했다.
- XAI도 별도로 train을 재분할하지 않고 명시적인 validation 데이터로 평가하도록 수정했다.
- LightGBM의 최대 boosting round를 500에서 3,000으로 늘리고, validation multi-logloss가 50 round 동안 개선되지 않으면 종료하도록 early stopping을 적용했다.
- 전체 accuracy 개선 가능성을 확인하기 위한 첫 실험으로 `class_weight="balanced"`를 `class_weight=None`으로 변경했다.

### 개선된 부분

- 학습 데이터의 loss만 확인하던 기존 방식과 달리 독립 validation loss로 최적 boosting iteration을 결정한다.
- test 데이터는 최종 평가에만 사용하므로 모델 선택 과정의 test leakage를 방지한다.
- 전처리 단계에서 validation/test 통계가 encoder와 scaler 학습에 유입되지 않는다.
- 자동 balanced 가중치 때문에 희소한 MITM 클래스를 과도하게 예측하던 현상이 줄어드는지 확인할 수 있다.

### 확인된 문제점 및 주의사항

- 기존 결과는 70/30 train/test 분할이고 이번 결과는 70/15/15 분할이므로, 동일한 test 샘플에 대한 완전한 일대일 비교는 아니다.
- `class_weight=None`은 전체 accuracy와 MITM precision을 높일 가능성이 있지만 MITM recall을 떨어뜨릴 수 있으므로 클래스별 지표를 함께 확인해야 한다.
- 트리 모델에는 MinMaxScaler가 필수적이지 않으며, 범주형 변수를 LabelEncoder 숫자로 처리하는 현재 방식도 추후 LightGBM native categorical 처리와 비교할 필요가 있다.

### 실행 상태

- 원본 데이터 재처리와 분할 검증 완료.
- 최초 학습 실행은 Conda 환경을 명시하지 않아 결과 생성 전에 중단했다.

## 2026-08-05 16:15 KST — Conda 실험 환경 확인

### 변경 사항

- 이후 전처리와 학습 명령은 셸에서 `conda activate toniot`를 수행한 뒤 실행하도록 통일했다.

### 확인된 문제점

- `toniot` Conda 환경 경로는 `/home/tako/anaconda3/envs/toniot`이지만 설치된 패키지 목록이 비어 있다.
- 환경 안에 `bin/python`이 없어 활성화 후에도 `/home/tako/anaconda3/bin/python`으로 fallback된다.
- 따라서 현재 상태의 `toniot`은 독립적으로 재현 가능한 실험 환경이 아니다. 환경에 Python과 프로젝트 의존성을 설치하거나 환경 정의 파일을 만드는 후속 조치가 필요하다.

### 현재 fallback 패키지 버전

- Python 실행 파일: `/home/tako/anaconda3/bin/python`
- LightGBM: 4.6.0
- scikit-learn: 1.8.0
- pandas: 3.0.1

### 실행 상태

- `toniot` 활성화 상태에서 원본 데이터 재처리 및 모델 학습을 다시 실행할 예정이다.

## 2026-08-05 16:18 KST — CPU 병렬화 및 GPU 환경 점검

### 변경 사항

- `n_jobs=-1`로 80개 논리 CPU를 모두 사용하던 학습은 첫 20 boosting round에도 도달하지 못해 중단했다.
- 과도한 thread 병렬화 오버헤드를 피하기 위해 LightGBM의 `n_jobs`를 8로 제한했다.
- 사용 가능한 것으로 안내받은 GPU 0번과 1번을 LightGBM 학습에 사용할 수 있는지 점검했다.

### 확인된 문제점

- `nvidia-smi`가 `Driver/library version mismatch`로 실패한다. 확인된 NVML library 버전은 535.309이다.
- 현재 LightGBM 4.6.0 빌드는 CUDA Tree Learner를 포함하지 않아 `device_type="cuda"` 사용 시 `CUDA Tree Learner was not enabled in this build` 오류가 발생한다.
- OpenCL 기반 `device_type="gpu"`도 `No OpenCL device found` 오류로 사용할 수 없다.
- 현재 단일 LightGBM 모델 학습에서는 GPU 0/1을 사용할 수 없으므로 CPU 8-thread로 실험을 계속한다.

### 후속 개선 조건

- NVIDIA kernel driver와 NVML library 버전을 일치시켜야 한다.
- `USE_CUDA=1`로 빌드된 LightGBM을 실제 Python이 설치된 Conda 환경에 구성해야 한다.
- LightGBM 단일 모델은 일반적으로 GPU 하나를 사용하므로 GPU 0은 단일 학습, GPU 1은 병렬 hyperparameter 실험에 배정하는 방식이 적합하다.

## 2026-08-05 16:20 KST — 22-feature baseline 실행 결과

### 실행 조건

- Conda 활성 환경 표시: `toniot`
- 실제 Python 실행 파일: `/home/tako/anaconda3/bin/python` (환경 내 Python 부재로 fallback)
- Feature 수: 22
- Class weight: 없음
- 최대 boosting round: 3,000
- Early stopping: validation multi-logloss 50 round
- CPU thread: 8
- XAI: 이번 baseline 비교에서는 실행하지 않음

### 학습 결과

- 실행 시간: 약 28초
- Best iteration: 227
- Best validation multi-logloss: 0.0585262
- 저장 모델 크기: 7.77 MB

### Validation 성능

| Metric | Value |
|---|---:|
| Accuracy | 0.980257 |
| Macro precision | 0.956866 |
| Macro recall | 0.959628 |
| Macro F1 | 0.958059 |

### Test 성능

| Metric | 기존 0727 (22 features, balanced) | 현재 (22 features, no weight) | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.977903 | 0.977920 | +0.000017 |
| Macro precision | 0.942487 | 0.953154 | +0.010667 |
| Macro recall | 0.967886 | 0.952816 | -0.015070 |
| Macro F1 | 0.952460 | 0.952795 | +0.000335 |

### 주요 클래스 변화

| Class/Metric | 기존 0727 | 현재 | 변화 |
|---|---:|---:|---:|
| Normal precision | 0.9981 | 0.9963 | -0.0018 |
| Normal recall | 0.9961 | 0.9983 | +0.0022 |
| Normal F1 | 0.9971 | 0.9973 | +0.0002 |
| MITM precision | 0.6225 | 0.7405 | +0.1180 |
| MITM recall | 0.9010 | 0.7452 | -0.1558 |
| MITM F1 | 0.7363 | 0.7429 | +0.0066 |

### 해석 및 문제점

- 전체 accuracy와 macro F1은 사실상 유지됐고 macro precision은 약 1.07%p 개선됐다.
- Normal recall은 99.61%에서 99.83%로 개선되어 정상 트래픽을 공격으로 판단하는 false positive가 감소했다.
- 자동 balanced 가중치를 제거하면서 MITM 과대 예측이 줄어 precision은 11.80%p 개선됐다.
- 반대로 실제 MITM을 놓치는 비율이 늘어 MITM recall이 15.58%p 하락했다. 침입 탐지 관점에서는 이 하락을 그대로 수용하기 어렵다.
- 기존 결과와 현재 결과의 test split이 다르므로 변화량은 방향성을 보는 참고값이며 완전히 통제된 비교는 아니다.
- 다음 실험에서는 `None`과 `balanced` 사이의 수동 MITM class weight를 validation macro F1과 MITM recall 기준으로 탐색해야 한다.

### 생성 결과

- Report: `reports/lightgbm_260805`
- Model: `models/lightgbm_toniot_classification.pkl`

## 2026-08-05 16:25 KST — MITM(class 4) 저성능 원인 분석

### 클래스 번호 확인

- `LabelEncoder`가 공격 유형 문자열을 알파벳순으로 정렬하므로 `mitm`이 class 4로 매핑된다.
- 숫자 4 자체가 성능에 영향을 주는 것은 아니다.

### 데이터 불균형

- Train의 MITM은 729건으로, 다른 각 공격 클래스 14,000건의 약 1/19이다.
- Normal 35,000건과 비교하면 약 1/48이다.
- 기존 `class_weight="balanced"`는 부족한 정보를 보충하는 것이 아니라 MITM 판정 경계를 넓혀 recall과 false positive 사이의 trade-off를 만든다.

### 패턴 다양성과 반복 데이터

- Train MITM 729건 중 고유한 22-feature 패턴이 728개로 거의 모두 다르다.
- Test MITM 157건 중 train과 완전히 같은 패턴은 0건이다.
- 비교 대상 클래스의 test 패턴이 train에 완전히 동일하게 존재하는 비율은 다음과 같다.
  - Backdoor: 92.10%
  - DDoS: 44.97%
  - DoS: 72.27%
  - Injection: 1.57%
  - MITM: 0.00%
  - Normal: 55.56%
  - Password: 27.23%
  - Ransomware: 99.73%
  - Scanning: 77.47%
  - XSS: 56.83%
- 따라서 random row split에서는 반복 패턴이 많은 클래스의 test 성능이 높게 측정되는 반면, 표본이 적고 거의 모두 고유한 MITM은 새로운 패턴을 일반화해야 한다.

### 클래스 간 feature overlap

- Test MITM의 5-nearest train 이웃 중 MITM이 과반인 샘플은 157건 중 101건뿐이다.
- 가장 가까운 train 샘플의 클래스도 MITM 113건 외에 DoS 13건, Password 13건, Normal 9건 등으로 겹친다.
- 현재 모델이 놓친 MITM 40건은 주로 Normal 13건, DoS 12건, Password 8건으로 분류됐다.
- 현재 모델은 MITM TP 117, FN 40, FP 41이며 precision 0.7405, recall 0.7452이다.

### 제거된 식별 feature의 영향

- 전처리에서 제거한 `src_ip`, `dst_ip`, `src_port`, `dst_port` 일부는 MITM 데이터에 매우 강하게 집중되어 있다.
- 예를 들어 `src_ip=192.168.1.34`인 528건 중 524건이 MITM이다.
- 이 값을 포함하면 현재 데이터셋 점수는 오를 가능성이 높지만 특정 장비/IP를 외우는 leakage가 되어 다른 네트워크로의 일반화 성능을 과대평가할 수 있다.

### 결론 및 후속 과제

- MITM 저성능의 핵심 원인은 class ID가 아니라 심한 표본 부족, 높은 패턴 다양성, 다른 클래스와의 feature overlap이다.
- 반복 행이 클래스별로 크게 다른 현재 random split은 공정한 일반화 성능 비교에 한계가 있다.
- 다음 단계는 중복 패턴을 같은 split에 묶는 group-aware split 검토, MITM 데이터 추가 확보, validation 기반 수동 class weight 탐색이다.

## 2026-08-05 17:08 KST — SAGE 중심 XAI 1차 축소 실험

### 계획 및 환경

- 확정된 구현 계획을 `XAIImprovementPlan_260805.md`에 별도로 저장했다.
- 비어 있던 `toniot` Conda 환경에 Python 3.11.15와 프로젝트 의존성을 설치했다.
- 사용자 전역 site-packages를 사용하지 않아도 실행되도록 환경 내부에 패키지를 설치하고 `PYTHONNOUSERSITE=1`로 검증했다.
- SAGE 구현을 위해 `sage-importance==0.0.6`을 설치했다.
- 재현 가능한 패키지 목록을 `requirements-toniot.txt`에 저장했다.

### XAI 코드 개선

- SAGE를 주 선택 기준으로 추가했다.
- SAGE는 train 147,729개 전체 행을 정확히 한 번씩 사용하는 full-dataset permutation 방식으로 계산했다.
- 제거된 피처는 train의 다른 행에서 가져온 marginal donor 값으로 대체하고 피처를 무작위 순서로 복원하면서 one-vs-rest binary cross-entropy 감소량을 누적했다.
- 각 클래스의 positive/negative loss를 50:50으로 균형화하고 10개 클래스 결과를 동일 비중으로 평균해 Macro-SAGE를 만들었다.
- SHAP은 기존 validation 최대 3,000건 대신 train 전체 147,729건을 batch 처리했다.
- CPF도 train 전체 147,729건과 22개 전체 피처를 사용해 피처별 5회 conditional permutation을 수행했다.
- SHAP interaction은 train batch에서 multiclass output별 tensor가 유효한 경우에만 파트너 보호에 사용하도록 유지했다.
- 선택 규칙을 Macro-SAGE top14 core, class-wise SHAP top12와 positive CPF top12 교집합 보호, Macro-SAGE 순위 기반 최소 18개 보충, interaction 기반 최대 20개로 변경했다.
- LIME은 1차 축소 자동 선택에서 제외했다.

### SHAP Interaction 제한

- 현재 LightGBM/SHAP 조합은 10-class 모델에 대해 단일 `(samples, features, features)` interaction tensor만 반환했다.
- 단일 tensor를 클래스별 결과로 복제하면 잘못된 설명이 되므로 이번 실험에서는 interaction 보호를 안전하게 생략했다.
- 따라서 이번 최종 18개는 SAGE core와 SHAP/CPF 클래스 보호로 결정됐다.

### 선택 결과

- 선택 피처: 18 / 22
- 선택된 피처:
  - `proto`, `service`, `duration`, `src_bytes`, `dst_bytes`, `conn_state`
  - `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`
  - `dns_query`, `dns_qclass`, `dns_qtype`, `dns_rcode`
  - `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`
- 제거된 피처:
  - `missed_bytes`
  - `http_request_body_len`
  - `http_response_body_len`
  - `http_status_code`
- 제거 피처의 Macro-SAGE 순위는 각각 19, 22, 21, 20위였다.

### Validation 채택 판정

| Metric | 22-feature baseline | 18-feature reduction-1 | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.980257 | 0.980131 | -0.000126 |
| Macro F1 | 0.958059 | 0.957370 | -0.000689 |
| MITM recall | 0.789809 | 0.783439 | -0.006369 |

- Accuracy 하락 0.0126%p: 허용 기준 0.1%p 이내
- Macro F1 하락 0.0689%p: 허용 기준 0.2%p 이내
- MITM recall 하락 0.6369%p: 허용 기준 1%p 이내
- 세 조건과 18~20개 피처 조건을 모두 만족하여 reduction-1을 채택하고 test를 평가했다.

### Test 결과

| Metric | 22-feature baseline | 18-feature reduction-1 | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.977920 | 0.978172 | +0.000253 |
| Macro precision | 0.953154 | 0.955566 | +0.002413 |
| Macro recall | 0.952816 | 0.956200 | +0.003385 |
| Macro F1 | 0.952795 | 0.955692 | +0.002897 |

### MITM 및 Normal 변화

| Class/Metric | 22-feature baseline | 18-feature reduction-1 | 변화 |
|---|---:|---:|---:|
| MITM precision | 0.7405 | 0.7625 | +0.0220 |
| MITM recall | 0.7452 | 0.7771 | +0.0319 |
| MITM F1 | 0.7429 | 0.7697 | +0.0268 |
| Normal precision | 0.9963 | 0.9960 | -0.0003 |
| Normal recall | 0.9983 | 0.9976 | -0.0007 |
| Normal F1 | 0.9973 | 0.9968 | -0.0005 |

- Test에서는 validation보다 좋은 방향으로 일반화되어 MITM precision, recall, F1이 모두 개선됐다.
- Normal 성능은 0.03~0.07%p 정도 소폭 하락했지만 여전히 F1 99.68%다.

### 효율 변화

- Best iteration: 227 -> 203
- 모델 크기: 7.77 MB -> 6.94 MB, 약 10.8% 감소
- 평균 추론 시간: 0.074248 ms -> 0.064510 ms, 약 13.1% 감소
- 처리량: 13,468 -> 15,501 samples/sec, 약 15.1% 증가
- 시간 지표는 서버 부하에 영향을 받으므로 동일 조건 반복 측정이 필요하다.

### 확인된 문제와 후속 개선

- full-train CPF의 22 feature x 5 repeat가 이번 실행에서 가장 긴 병목이었다.
- 최초 실행에는 CPF 피처별 진행률이 없어 장시간 무출력 상태였으며, 이후 실행을 위해 피처별 진행 로그를 추가했다.
- SAGE는 전체 행 수가 충분하다고 판단해 1 repeat로 실행했으므로 repeat 간 불확실성 값은 0이다. 안정성 검증이 필요하면 seed를 바꾼 독립 실행을 추가해야 한다.
- 현재 multiclass SHAP interaction은 유효하지 않아 파트너 보호가 작동하지 않았다. 추후 one-vs-rest interaction 모델이나 검증된 multiclass interaction 방법을 별도 검토해야 한다.
- 전체 train을 사용한 결과는 모델의 학습 동작을 설명한다. 축소 피처의 일반화 여부는 이번처럼 validation 채택 기준과 test 최종 평가를 분리해 확인해야 한다.

### 생성 결과

- 전체 실행 로그: `260805.txt`
- XAI 결과: `reports/lightgbm_260805/xai/reduction_1`
- 축소 모델 결과: `reports/lightgbm_260805/reduction_1`
- 축소 모델: `models/lightgbm_toniot_classification_reduction_1.pkl`

## 2026-08-05 17:29 KST — SAGE 중심 XAI 2차 축소 실험

### 보존 및 실행 조건

- reduction-1에서 선택한 18개 피처만 reduction-2의 입력으로 사용했다.
- reduction-1 보고서, XAI 결과, 모델, `260805.txt`의 실행 전후 SHA-256 checksum이 모두 일치함을 확인했다.
- reduction-2 전용 경로와 모델명을 사용해 이전 결과를 덮어쓰지 않았다.
- class weight, LightGBM hyperparameter, train/validation/test 분할은 이전 실험과 동일하게 유지했다.

### 2차 선택 설정

- 입력 피처: 18개
- Macro-SAGE core: top 12
- Class-wise SHAP: top 10
- Positive class-wise CPF: top 10
- 클래스별 최소 보호 피처: 3개
- 목표 최종 피처: 16~17개
- SHAP interaction partner: 유효할 때 최대 17개까지 허용
- LIME: 자동 선택에서 제외

### SHAP Interaction 제한

- reduction-2에서도 multiclass SHAP interaction이 클래스별 10개 tensor가 아닌 단일 tensor만 반환됐다.
- 잘못된 클래스별 복제를 하지 않고 interaction 보호를 안전하게 생략했다.

### 선택 결과

- 선택 피처: 16 / 18
- reduction-1에서 추가 제거된 피처:
  - `dns_qclass` — Macro-SAGE 18위
  - `dns_qtype` — Macro-SAGE 17위
- 최종 피처:
  - `proto`, `service`, `duration`, `src_bytes`, `dst_bytes`, `conn_state`
  - `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`
  - `dns_query`, `dns_rcode`, `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`

### Validation 채택 판정

| Metric | Reduction-1 18개 | Reduction-2 16개 | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.980131 | 0.980320 | +0.000190 |
| Macro F1 | 0.957370 | 0.958986 | +0.001616 |
| MITM recall | 0.783439 | 0.796178 | +0.012739 |

- 세 핵심 validation 지표가 모두 개선됐다.
- 피처 수 조건도 만족하여 reduction-2를 채택하고 test를 평가했다.

### Test 결과 비교

| Metric | 22개 baseline | Reduction-1 18개 | Reduction-2 16개 |
|---|---:|---:|---:|
| Accuracy | 0.977920 | 0.978172 | 0.978204 |
| Macro precision | 0.953154 | 0.955566 | 0.954429 |
| Macro recall | 0.952816 | 0.956200 | 0.956857 |
| Macro F1 | 0.952795 | 0.955692 | 0.955419 |

- 22개 baseline과 비교하면 reduction-2도 accuracy와 Macro F1이 개선됐다.
- reduction-1과 비교하면 accuracy는 0.0032%p, Macro recall은 0.0657%p 증가했다.
- 반면 Macro precision은 0.1137%p, Macro F1은 0.0272%p 감소했다.

### MITM 및 Normal 비교

| Class/Metric | Reduction-1 18개 | Reduction-2 16개 | 변화 |
|---|---:|---:|---:|
| MITM precision | 0.7625 | 0.7500 | -0.0125 |
| MITM recall | 0.7771 | 0.7834 | +0.0063 |
| MITM F1 | 0.7697 | 0.7664 | -0.0033 |
| Normal precision | 0.9960 | 0.9961 | +0.0001 |
| Normal recall | 0.9976 | 0.9975 | -0.0001 |
| Normal F1 | 0.9968 | 0.9968 | 동일 |

- reduction-2는 MITM을 한 건 더 탐지해 recall이 증가했지만 false positive도 늘어 precision과 F1은 소폭 낮아졌다.

### 효율 비교 및 해석

- Best iteration: reduction-1 203 -> reduction-2 219
- 모델 파일 크기: 7,273,284 -> 7,853,780 bytes, 약 8.0% 증가
- 평균 추론 시간: 0.064510 -> 0.068554 ms, 약 6.3% 증가
- 처리량: 15,501 -> 14,587 samples/sec, 약 5.9% 감소
- 피처는 2개 줄었지만 더 많은 boosting tree가 필요해 모델 크기와 실측 추론 성능은 reduction-1보다 불리했다.
- 종합 성능, MITM F1, 모델 크기와 속도를 함께 고려하면 reduction-1 18개 모델을 기본 추천한다.
- 피처 수 자체를 최소화하거나 MITM recall을 조금 더 우선하면 reduction-2 16개 모델도 채택 가능한 대안이다.

### 평가상 주의사항

- reduction-1 test 결과를 본 뒤 reduction-2를 시작했으므로 test가 더 이상 완전히 독립적인 최초 holdout이라고 보기는 어렵다.
- 최종 배포 모델을 공정하게 확정하려면 별도 untouched holdout 또는 group-aware split에서 한 번 더 비교해야 한다.

### 생성 결과

- 전체 실행 로그: `260805_reduction_2.txt`
- XAI 결과: `reports/lightgbm_260805/xai/reduction_2`
- 축소 모델 결과: `reports/lightgbm_260805/reduction_2`
- 축소 모델: `models/lightgbm_toniot_classification_reduction_2.pkl`

## 2026-08-05 17:35 KST — 전체 파이프라인 다이어그램 작성

### 변경 사항

- 데이터 전처리부터 reduction-2까지의 전체 학습·XAI·피처 선택 과정을 하나의 SVG 다이어그램으로 작성했다.
- `figs/` 디렉터리를 새로 만들고 다이어그램과 Markdown 미리보기 문서를 저장했다.

### 다이어그램 포함 범위

- 원본 211,043행 전처리와 train/validation/test 분할
- 22-feature LightGBM baseline 학습 및 early stopping
- train 전체 기반 Macro-SAGE, class-wise SHAP, class-wise CPF
- multiclass SHAP interaction 안전 생략 조건과 LIME 역할
- reduction-1 22 -> 18, reduction-2 18 -> 16 선택 규칙
- validation acceptance gate, test 평가, 모델·보고서 보존
- reduction-1과 reduction-2의 핵심 성능 및 최종 사용 권고

### 생성 결과

- SVG: `figs/toniot_training_feature_extraction_pipeline.svg`
- PNG: `figs/toniot_training_feature_extraction_pipeline.png`
- 미리보기: `figs/README.md`
- SVG XML parsing 및 크기 검증을 통과했다.
- headless browser로 PNG 렌더링 후 전체 레이아웃을 확인했으며 내용 잘림이나 박스 겹침이 없음을 확인했다.

## 2026-08-05 17:38 KST — Reduction-3 실험 시작

### 변경 사항

- reduction-2의 채택 여부와 16개 고유 피처를 확인한 후에만 reduction-3가 실행되도록 입력 검증을 추가했다.
- reduction-3의 목표를 14~15개 피처로 설정했다.
- Macro-SAGE top 10, class-wise SHAP top 8, CPF top 8, 클래스별 최소 보호 피처 3개를 적용한다.
- reduction-2 대비 validation accuracy, macro F1, MITM recall 하락 기준을 모두 만족할 때만 test를 평가한다.
- GPU 0/1 사용 가능 여부를 확인했으나 NVIDIA driver 통신 실패로 기존과 동일한 CPU 8-thread 환경을 유지한다.

### 저장 위치

- 코드 진입점: `run_reduction_3.py`
- XAI 결과: `reports/lightgbm_260805/xai/reduction_3`
- 모델 결과: `reports/lightgbm_260805/reduction_3`
- 전체 실행 로그: `260805_reduction_3.txt`

## 2026-08-05 17:55 KST — Reduction-3 실험 결과

### 선택 결과

- reduction-2의 16개 중 14개를 선택했다.
- 제거된 피처: `dns_rcode`, `dns_RD`
- 선택된 피처: `proto`, `service`, `duration`, `src_bytes`, `dst_bytes`, `conn_state`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`, `dns_query`, `dns_AA`, `dns_RA`, `dns_rejected`
- SHAP interaction은 10-class별 tensor가 아닌 단일 tensor만 반환되어 잘못 복제하지 않고 안전하게 생략했다.

### Validation 비교

| 지표 | Reduction-2 (16) | Reduction-3 (14) | 하락폭 | 기준 | 결과 |
|---|---:|---:|---:|---:|---|
| Accuracy | 98.0320% | 97.9941% | 0.0379%p | ≤ 0.1%p | 통과 |
| Macro F1 | 95.8986% | 95.7218% | 0.1767%p | ≤ 0.2%p | 통과 |
| MITM recall | 79.6178% | 78.3439% | 1.2739%p | ≤ 1.0%p | 실패 |

### 결론 및 문제점

- reduction-3는 MITM recall 보호 기준을 0.2739%p 초과했으므로 기각했다.
- Validation gate 실패에 따라 test 평가는 수행하지 않았으며 reduction-3 모델도 저장하지 않았다.
- XAI 결과, 선택 목록, validation 비교, 채택 결정, 실행 로그는 재현과 분석을 위해 보존했다.
- reduction-3 진행 기준으로 마지막 채택 단계는 `models/lightgbm_toniot_classification_reduction_2.pkl`의 16-feature 모델이다. 다만 속도·크기·MITM F1을 함께 본 기존 종합 추천은 reduction-1이며, reduction-2는 최소 피처/MITM recall 우선 대안이다.
- 전체 파이프라인 SVG/PNG에도 reduction-3 선택과 validation 기각 경로를 추가했다.
- 자동 생성 XAI README의 단계 고정 문구(`reduction-1`)를 모든 단계에 맞는 일반 문구로 수정했다.

## 2026-08-05 18:17 KST — 원본과 Reduction-1 비교 문서 작성

### 생성 결과

- `260805_baseline_vs_reduction_1.txt`에 원본 22-feature 모델과 Reduction-1 18-feature 모델의 비교를 작성했다.
- 비교 범위: feature 구성, validation/test 전체 지표, MITM/Normal 클래스, 모델 크기, best iteration, 추론 시간 및 처리량

### 핵심 비교

- Reduction-1은 feature 수를 22개에서 18개로 18.18% 줄였다.
- Test Accuracy는 0.0253%p, Macro F1은 0.2897%p, MITM F1은 2.68%p 상승했다.
- 모델 크기는 10.78%, 평균 추론 시간은 13.12% 감소했다.
- Normal F1은 0.05%p 하락했지만 99.68%를 유지했다.
- 현재 split 기준으로 Reduction-1이 원본보다 성능 균형과 효율이 모두 우수하다고 판단했다.
