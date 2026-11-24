<p align="center">
  <img src="docs/assets/hero.svg" alt="TON IoT IDS — XAI 특징 선택과 LightGBM 이상행위 탐지" width="1000">
</p>

<p align="center">
  <b>한국어</b> · <a href="README.md">English</a> ·
  <a href="https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실험-결과">실험 결과</a> ·
  <a href="https://github.com/M3rcy1028/zt-iot-security">통합 보안 시스템</a>
</p>

## IoT 네트워크를 위한 설명 가능한 이상행위 탐지

TON_IoT 네트워크 데이터로 LightGBM을 학습하고 **Macro-SAGE·class-wise SHAP·approximate CPFI**를 결합해 공격 유형별 주요 feature를 보존하면서 모델 입력을 줄인다. 학습 모델은 [제로 트러스트 IoT 보안 시스템](https://github.com/M3rcy1028/zt-iot-security)의 내부 서버에서 장치별 접근제어에 필요한 분석 결과를 제공한다.

| 확인할 내용 | 링크 |
| --- | --- |
| 한국어 실험 안내 | [전체 README.ko.md](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md) |
| English experiment guide | [Full README](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.md) |
| 데이터셋·분할·유형별 분포 | [데이터셋](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#데이터셋) |
| 학습 재현·모델 실행 | [실행 방법](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실행-방법) |
| 실험 결과 요약 | [Research summary](https://github.com/M3rcy1028/toniot-ids/blob/xai/docs/research_summary.json) |
| 배포 모델 | [18-feature LightGBM](https://github.com/M3rcy1028/toniot-ids/blob/xai/models/lightgbm_toniot_classification_reduction_1.pkl) |

```bash
git clone --branch xai --single-branch --depth 1 https://github.com/M3rcy1028/toniot-ids.git
cd toniot-ids
```

[환경 구성과 실행 절차](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실행-방법)에 따라 실행한다. 의존성 버전은 `requirements-toniot.txt`로 관리하며 모델 로딩·추론 환경은 CPython 3.14.4이다.

## 데이터와 주요 결과

본 실험은 [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) 네트워크 CSV **211,043행·44열·10개 유형**을 사용한다. Stratified split으로 train·validation·test set을 **147,729 / 31,657 / 31,657건**으로 구성하였다.

F1은 10개 클래스의 macro average이다. (분류 지표 단위: %)

| 모델 | Feature 수 | Test Accuracy | Test Macro F1 | 모델 크기 (B) | Batch time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기준 모델 | 22 | 97.7920 | 95.2795 | 8,152,020 | 2.3505 |
| **배포 모델** | **18** | **97.8172** | **95.5692** | **7,273,284** | **2.0422** |
| 추가 축소 | 16 | 97.8204 | 95.5419 | 7,853,780 | 2.1702 |

![분류 성능과 자원 비용 비교](docs/assets/model_comparison.png)

18-feature 모델은 baseline보다 입력 수가 **18.18%**, 모델 크기가 **10.78%**, batch time이 **13.12%** 감소하였고 Macro F1은 0.2897%p 증가하였다. 16-feature 모델의 Accuracy가 조금 더 높았으나, 통합 시스템에서는 Macro F1·모델 크기·처리 시간을 비교해 18-feature 모델을 선정하였다. Batch time은 31,657건에 대한 **`predict`와 `predict_proba` 두 호출의 합계**이다.

유형별 성능·confusion matrix·feature importance·validation 결과는 [상세 실험 결과](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#실험-결과)에 정리하였다.

## 연계 자료

- [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets): 데이터셋 설명과 인용 논문
- [학습·평가 문서](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md): 전처리·XAI·모델·실험 결과
- [제로 트러스트 IoT 보안 시스템](https://github.com/M3rcy1028/zt-iot-security): 인증 통신·서버 분석·장치별 위험 기반 접근제어

## 라이선스

프로젝트 코드와 문서는 [MIT License](LICENSE)를 따른다. 데이터셋과 외부 라이브러리의 이용 조건은 [Third-party notices](THIRD_PARTY_NOTICES.md)에 정리하였다.

## 고려 외 사항

- 네트워크 CSV의 오프라인 분류·행 단위 stratified split 사용
- 실시간 packet capture·원시 센서 telemetry 탐지·외부 환경 평가 미수행
- Batch time에 feature 추출·통신 시간 미포함, 단일 요청 latency 미측정
- 동일 test set에서 후보 모델 비교 후 독립적인 external test 미수행
- 14-feature 후보는 validation에서 미채택, test 지표·모델 파일 미생성
- 원 학습 환경의 정확한 Python 버전·하드웨어 기록 미비
- [전체 실험의 고려 외 사항](https://github.com/M3rcy1028/toniot-ids/blob/xai/README.ko.md#고려-외-사항)
