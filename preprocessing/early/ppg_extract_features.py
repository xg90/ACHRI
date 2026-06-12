import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_signal_from_xlsx(path: Path, column_index: int | None) -> np.ndarray:
    df = pd.read_excel(path)
    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {path}")

    if column_index is None:
        # Use the only column if there is one, otherwise default to the second column.
        idx = 0 if df.shape[1] == 1 else 1
    else:
        idx = column_index

    if idx < 0 or idx >= df.shape[1]:
        raise ValueError(f"Column index {idx} out of range for {path} (cols={df.shape[1]})")

    series = pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna()
    return series.to_numpy(dtype=float)


def split_segments(signal: np.ndarray, segment_len: int, segment_count: int) -> list[np.ndarray]:
    total_needed = segment_len * segment_count
    if len(signal) < total_needed:
        raise ValueError(
            f"Signal too short: need {total_needed} samples, got {len(signal)}"
        )
    trimmed = signal[:total_needed]
    return [trimmed[i * segment_len:(i + 1) * segment_len] for i in range(segment_count)]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract PPG features from baseline-subtracted XLSX files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(script_dir / "PPG2"),
        help="Folder containing 1.xlsx .. 16.xlsx",
    )
    parser.add_argument(
        "--output",
        default="PPG1.xlsx",
        help="Output Excel file name (saved in input-dir if relative)",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=5000,
        help="Samples per segment",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=6,
        help="Segments per file",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=250.0,
        help="Sample rate (Hz) passed to HeartPy",
    )
    parser.add_argument(
        "--column-index",
        type=int,
        default=-1,
        help="0-based column index to use; -1 means auto (only column or 2nd column).",
    )
    parser.add_argument(
        "--ppg-code-dir",
        default=str(script_dir / "ppg_feature_extraction" / "ppg_feature_extraction"),
        help="Directory that contains process_ppg.py",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_dir / output_path

    column_index = None if args.column_index == -1 else args.column_index

    ppg_code_dir = Path(args.ppg_code_dir)
    if not (ppg_code_dir / "process_ppg.py").exists():
        raise FileNotFoundError(f"process_ppg.py not found in: {ppg_code_dir}")

    sys.path.insert(0, str(ppg_code_dir))
    import process_ppg  # noqa: E402

    rows = []
    files = sorted(
        [p for p in input_dir.glob("*.xlsx") if p.name.lower() != output_path.name.lower()]
    )
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {input_dir}")

    feature_cols = [
        "bpm",
        "ibi",
        "sdnn",
        "sdsd",
        "rmssd",
        "pnn20",
        "pnn50",
        "hr_mad",
        "sd1",
        "sd2",
        "s",
        "sd1/sd2",
        "breathingrate",
    ]

    for file_path in files:
        signal = load_signal_from_xlsx(file_path, column_index=column_index)
        segments = split_segments(signal, args.segment_len, args.segment_count)

        for idx, segment in enumerate(segments, start=1):
            row = {
                "file": file_path.name,
                "segment": idx,
            }
            try:
                _filtered, _wd, measures, extraction_mode = process_ppg.run_heartpy_with_fallback(
                    segment, sample_rate=args.sample_rate
                )
                for col in feature_cols:
                    row[col] = measures.get(col, "")
                row["extraction_mode"] = extraction_mode
            except Exception:
                for col in feature_cols:
                    row[col] = ""
                row["extraction_mode"] = "analysis_failed"
            rows.append(row)

    df_out = pd.DataFrame(
        rows,
        columns=["file", "segment"] + feature_cols + ["extraction_mode"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(output_path, index=False)
    print(f"Saved: {output_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
