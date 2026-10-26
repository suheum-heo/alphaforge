"""Exercise a real local server using fresh CSV data, then shut it down."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from alphaforge.deep import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "alphaforge.cli",
            "serve",
            "--bundle",
            str(args.bundle),
            "--port",
            str(args.port),
        ]
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("API server exited before becoming ready")
            try:
                with urlopen(base + "/health", timeout=1) as response:
                    assert response.status == 200
                break
            except (URLError, TimeoutError):
                time.sleep(0.1)
        else:
            raise RuntimeError("API did not become ready")
        predictor = Predictor(args.bundle)
        frame = pd.read_csv(args.input)
        frame = frame.loc[frame.symbol == predictor.metadata["symbol"]]
        payload = json.dumps({"candles": frame.to_dict("records")}).encode()
        request = Request(
            base + "/predict", data=payload, headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        offline = predictor.predict_candles(frame)
        assert abs(result["predicted_log_return"] - offline["predicted_log_return"]) < 1e-8
        evidence = {"http_status": 200, "offline_match": True, "prediction": result}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
