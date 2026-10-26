"""AlphaForge data, training, walk-forward evaluation and inference commands."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .data import ingest, synthetic
from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alphaforge")
    parser.add_argument(
        "command",
        choices=[
            "ingest",
            "run",
            "demo",
            "train",
            "backtest",
            "predict",
            "serve",
            "study",
            "dashboard",
        ],
    )
    parser.add_argument("--config", type=Path, help="TOML configuration; defaults if omitted")
    parser.add_argument("--input", type=Path, help="CSV candles")
    parser.add_argument("--output", type=Path, help="New output file/directory")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--bundle", type=Path, help="Saved model directory for predict/serve")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--model",
        default="ridge",
        choices=["zero_return", "training_mean", "ridge", "lstm", "transformer"],
    )
    parser.add_argument("--initial-train-hours", type=int, default=2160)
    parser.add_argument("--validation-hours", type=int, default=336)
    parser.add_argument("--test-hours", type=int, default=336)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = Config.load(args.config) if args.config else Config()
        if args.command == "dashboard":
            from .dashboard import export_dashboard

            print(json.dumps(export_dashboard(args.runs, args.output or Path("docs")), indent=2))
            return
        if args.command in ("predict", "serve"):
            if args.bundle is None:
                parser.error("predict/serve requires --bundle")
            if args.command == "serve":
                import uvicorn

                from .api import create_app

                uvicorn.run(create_app(args.bundle), host="127.0.0.1", port=args.port)
                return
            if args.input is None:
                parser.error("predict requires --input")
            from .deep import Predictor

            predictor = Predictor(args.bundle)
            candles = pd.read_csv(args.input)
            candles = candles.loc[candles.symbol == predictor.metadata["symbol"]]
            result = predictor.predict_candles(candles)
            print(json.dumps(result, indent=2))
            return
        if args.output is None:
            parser.error("This command requires --output")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new path")
        if args.command == "ingest":
            frame = ingest(config)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                frame.to_csv(handle, index=False)
            print(f"Saved {len(frame)} closed candles to {args.output}")
            return
        if args.command != "demo" and args.input is None:
            parser.error("This command requires --input")
        frame = synthetic(config) if args.command == "demo" else pd.read_csv(args.input)
        source = "synthetic_demo" if args.command == "demo" else f"csv:{args.input.resolve()}"
        if args.command in ("train", "backtest", "study"):
            from .deep import TrainingConfig
            from .research import BacktestConfig, train, walk_forward

            settings = TrainingConfig(
                sequence_length=args.sequence_length,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                patience=args.patience,
            )
            if args.command == "study":
                from .experiment import study

                report = study(frame, config, settings, args.output)
            elif args.command == "train":
                report = train(frame, config, settings, args.output)
            else:
                execution = BacktestConfig(
                    args.initial_train_hours,
                    args.validation_hours,
                    args.test_hours,
                    args.fee_bps,
                    args.slippage_bps,
                    args.threshold,
                )
                report = walk_forward(frame, config, settings, execution, args.output, args.model)
        else:
            report = run(frame, config, args.output, source)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "source": source,
                    "results": report.get("results", report.get("strategy")),
                },
                indent=2,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"alphaforge: {exc}\n")
    except ImportError as exc:
        parser.exit(
            1,
            f"alphaforge: install optional dependencies with pip install -e "
            f'".[deep,api]" ({exc})\n',
        )


if __name__ == "__main__":
    main()
