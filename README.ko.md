<p align="center">
  <img src="docs/assets/hero.svg" alt="TON IoT IDS — XAI 특징 선택과 LightGBM 이상행위 탐지" width="1000">
</p>

<p align="center">
  <b>한국어</b> · <a href="README.md">English</a> ·
  <a href="https://github.com/M3rcy1028/toniot-ids/tree/xai">실험 브랜치</a> ·
  <a href="https://github.com/M3rcy1028/zt-iot-security">통합 보안 시스템</a>
</p>

## IoT 네트워크를 위한 설명 가능한 이상행위 탐지

TON_IoT 네트워크 특징으로 LightGBM을 학습하고 **Macro-SAGE·클래스별 SHAP·근사 CPFI**를 결합하여 공격 유형별 식별 정보를 보존하면서 입력 특징을 축소하는 연구이다. 학습된 모델은 [제로 트러스트 IoT 보안 시스템](https://github.com/M3rcy1028/zt-iot-security)의 내부 서버에서 장치별 접근 판단에 분석 결과를 제공한다.

### `xai` 브랜치에서 시작하기

**현재 학습 코드·모델·데이터 설명·상세 실험 결과는 [`xai`](https://github.com/M3rcy1028/toniot-ids/tree/xai)에 있다.** 기본 `main`은 이전 구현을 보존한다. 아래 수치는 `xai`의 저장 결과를 소개한 것으로 이전 `main` 코드의 평가 결과와 구분한다.

| 확인할 내용 | 링크 |
| --- | --- |
| 한국어 실험 안내 | [전체 README.ko.md](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md) |
| English experiment guide | [Full README](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.md) |
| 데이터셋·분할·유형별 분포 | [데이터셋](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#데이터셋) |
| 학습 재현·저장 모델 실행 | [실행 방법](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실행-방법) |
| 기계 판독 가능한 실험 근거 | [Research summary](https://github.com/M3rcy1028/toniot-ids/blob/xai/docs/research_summary.json) |
| 배포 모델 | [18개 특징 LightGBM 앙상블](https://github.com/M3rcy1028/toniot-ids/blob/xai/models/lightgbm_toniot_classification_reduction_1.pkl) |

```bash
git clone --branch xai --single-branch --depth 1 https://github.com/M3rcy1028/toniot-ids.git
cd toniot-ids
```

이후 해당 브랜치의 [환경 구성과 실행 절차](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실행-방법)를 따른다. 패키지 버전은 `requirements-toniot.txt`에 고정되어 있으며 모델 로딩·추론은 CPython 3.14.4에서 확인하였다. 기존 학습 실행의 정확한 Python 버전과 전체 장비 설정은 완전히 기록되어 있지 않다.

## 데이터와 주요 결과

본 저장소의 TON_IoT 네트워크 CSV **211,043행·44열**을 사용한다. 10개 유형을 학습·검증·테스트 **147,729 / 31,657 / 31,657건**으로 계층 분할하였다. 이 규모는 해당 CSV의 표본 수이며 [TON_IoT 전체 자료](https://research.unsw.edu.au/projects/toniot-datasets)의 규모를 의미하지 않는다.

아래 분류 지표는 %이며 F1은 10개 클래스의 비가중 Macro 평균이다.

| 모델 | 특징 수 | 테스트 Accuracy | 테스트 Macro F1 | 모델 크기 (B) | 배치 시간 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기준 모델 | 22 | 97.7920 | 95.2795 | 8,152,020 | 2.3505 |
| **배포 모델** | **18** | **97.8172** | **95.5692** | **7,273,284** | **2.0422** |
| 추가 축소 | 16 | 97.8204 | 95.5419 | 7,853,780 | 2.1702 |

![기존 분류 성능과 자원 비용 비교](docs/assets/model_comparison.png)

18개 모델은 기준 모델보다 입력 수가 **18.18%**, 모델 파일 크기가 **10.78%**, 기록된 배치 평가 시간이 **13.12%** 감소하였고 Macro F1은 0.2897%p 증가하였다. 16개 모델의 Accuracy는 조금 더 높지만, 통합 시스템에서는 Macro F1·저장 크기·해당 실행의 처리 시간을 함께 고려하여 18개를 사용하였다. 14개 후보는 검증 MITM Recall 조건을 통과하지 못해 테스트 결과를 저장하지 않았다. 시간은 31,657건에 대한 **`predict`와 `predict_proba` 두 호출의 합계**이며 단일 요청 또는 장치–서버 통신 지연과 구분한다.

위 값은 [`c8cc8596`](https://github.com/M3rcy1028/toniot-ids/tree/c8cc8596f22900d994eca525878c8c674e2e98e4)의 기존 실험 자료로, 예측 파일·혼동행렬과 수치를 대조하였다. 낮은 MITM 성능, 검증 판단, 원본 그림과 평가 범위는 [상세 결과](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실험-결과)에 함께 정리하였다.

## 연계 자료

- [TON_IoT 원 제공처](https://research.unsw.edu.au/projects/toniot-datasets): 원본 설명·논문·필수 인용 자료.
- [`xai` 브랜치](https://github.com/M3rcy1028/toniot-ids/tree/xai): 모델 개발·전처리·XAI·평가 자료.
- [`zt-iot-security`](https://github.com/M3rcy1028/zt-iot-security): 장치 인증·암호화 통신·내부 서버 분석·장치별 위험 기반 접근제어.
