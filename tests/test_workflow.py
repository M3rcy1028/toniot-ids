"""Exercise the public CLI with isolated test data and real LightGBM fitting."""
from pathlib import Path
import csv
import os
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("backdoor", "ddos", "dos", "injection", "mitm", "normal", "password", "ransomware", "scanning", "xss")


class WorkflowTest(unittest.TestCase):
    def run_cli(self, workspace, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "toniot_ids", "--workspace", str(workspace), *arguments],
            cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
            env={**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"},
            timeout=120,
        )

    def test_preprocessing_training_and_model_serialization(self):
        with tempfile.TemporaryDirectory(prefix="toniot-workflow-") as directory:
            workspace = Path(directory).resolve()
            self.assertEqual(workspace.parent, Path(tempfile.gettempdir()).resolve())
            # Constant features make early stopping fast while exercising all ten labels.
            features = dict(
                proto="tcp", service="none", duration=0, src_bytes=1, dst_bytes=1,
                conn_state="S0", missed_bytes=0, src_pkts=1, src_ip_bytes=20,
                dst_pkts=1, dst_ip_bytes=20, dns_query="none", dns_qclass=0,
                dns_qtype=0, dns_rcode=0, dns_AA="F", dns_RD="F", dns_RA="F",
                dns_rejected="F", http_request_body_len=0, http_response_body_len=0,
                http_status_code=0,
            )
            raw = workspace / "data_network/raw/train_test_network.csv"
            raw.parent.mkdir(parents=True)
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[*features, "src_ip", "dst_ip", "src_port", "dst_port", "label", "type"])
                writer.writeheader()
                for label in CLASSES:
                    for _ in range(20):
                        writer.writerow({**features, "src_ip": "source", "dst_ip": "destination",
                                         "src_port": 1, "dst_port": 2, "label": int(label != "normal"), "type": label})

            prepared = self.run_cli(workspace, "preprocess")
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            processed = workspace / "data_network/processed_type"
            sizes = [len(pd.read_csv(processed / f"X_{s}.csv")) for s in ("train", "valid", "test")]
            self.assertEqual(sum(sizes), 200)
            for actual, expected in zip(sizes, (140, 30, 30), strict=True):
                self.assertLessEqual(abs(actual - expected), 1)

            trained = self.run_cli(workspace, "baseline", "--no-xai")
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            model = joblib.load(workspace / "models/lightgbm_toniot_classification.pkl")
            target = joblib.load(processed / "target_encoder.pkl")
            test = pd.read_csv(processed / "X_test.csv")
            self.assertEqual(tuple(target.classes_), CLASSES)
            self.assertEqual(model.n_features_in_, 22)
            self.assertEqual(model.n_classes_, 10)
            probabilities = model.predict_proba(test)
            self.assertEqual(probabilities.shape, (30, 10))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)
            reports = list((workspace / "reports").glob("lightgbm_*"))
            self.assertEqual(len(reports), 1)
            predictions = pd.read_csv(reports[0] / "predictions.csv")
            np.testing.assert_array_equal(predictions["y_pred"], model.predict(test))
            self.assertFalse((reports[0] / "xai").exists())

    def test_missing_feature_list_stops_before_training(self):
        with tempfile.TemporaryDirectory(prefix="toniot-workflow-") as directory:
            workspace = Path(directory).resolve()
            self.assertEqual(workspace.parent, Path(tempfile.gettempdir()).resolve())
            result = self.run_cli(workspace, "baseline", "--features", "missing.csv", "--no-xai")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Feature list not found", result.stderr)
            self.assertEqual(list(workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
