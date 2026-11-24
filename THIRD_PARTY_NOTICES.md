# Third-party notices

The [MIT License](LICENSE) covers this project's original contributions. It does not replace the terms of third-party datasets or dependencies.

## TON_IoT data

The dataset copies and derived preprocessing files in `data/`, `data_network/`, and `data_iot/` originate from the UNSW TON_IoT collection. The network download code uses a Kaggle mirror.

- Original provider: [UNSW TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)
- Network mirror: [TON_IoT Network Dataset on Kaggle](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset)
- The provider grants academic research use and requires permission for commercial use
- Follow the provider's publication and citation requirements

The repository's MIT license does not relicense these datasets.

## Dependencies and reference implementations

Dependencies are installed separately and retain their upstream licenses and copyright notices.

| Project | Upstream source |
| --- | --- |
| LightGBM | [microsoft/LightGBM](https://github.com/microsoft/LightGBM) |
| scikit-learn | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) |
| pandas | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| Matplotlib | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) |
| joblib | [joblib/joblib](https://github.com/joblib/joblib) |
| SHAP | [shap/shap](https://github.com/shap/shap) |
| SAGE | [iancovert/sage](https://github.com/iancovert/sage) |
| LIME | [marcotcr/lime](https://github.com/marcotcr/lime) |
| kagglehub | [Kaggle/kagglehub](https://github.com/Kaggle/kagglehub) |

The class-wise aggregation, grouped-shuffle CPFI, and consensus selection procedure are implemented in this repository's `xai.py`.
