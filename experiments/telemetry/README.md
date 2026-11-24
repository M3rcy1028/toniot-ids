# TON_IoT telemetry preprocessing

[`preprocess.py`](preprocess.py) contains the sensor-dataset preprocessing experiment. It reads the seven device CSVs from `data_iot/raw/`, writes per-device splits to `data_iot/processed_iot/`, and writes analysis reports to `reports/ton_iot_analysis/`.

```bash
python experiments/telemetry/preprocess.py
```

The LightGBM network experiment uses `data_network/` and the `python -m toniot_ids` CLI.
