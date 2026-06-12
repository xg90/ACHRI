import os
import re
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent
batch = str(script_dir / "PPG" / "batch2")

# 读取所有文件名
files = os.listdir(batch)

# 计算每个 bl 的 baseline 平均值
baseline = {}
num_re = re.compile(r"^(\d+)")
for f in files:
    if not f.lower().endswith("bl.csv"):
        continue
    m = num_re.match(f)
    if not m:
        continue
    num = int(m.group(1))
    path = os.path.join(batch, f)
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if c.strip().lower() == "ppg finger":
            col = c
            break
    if col is None:
        raise ValueError(f"PPG Finger column not found in {f}")
    baseline[num] = df[col].mean()

# 处理 s.csv 并保存为 1.xlsx~16.xlsx
for f in files:
    if not f.lower().endswith("s.csv"):
        continue
    m = num_re.match(f)
    if not m:
        continue
    num = int(m.group(1))
    if num not in baseline:
        raise ValueError(f"No baseline found for {num}")
    path = os.path.join(batch, f)
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if c.strip().lower() == "ppg finger":
            col = c
            break
    if col is None:
        raise ValueError(f"PPG Finger column not found in {f}")

    out_df = df[[col]].copy()
    out_df[col] = out_df[col] - baseline[num]

    out_path = os.path.join(batch, f"{num}.xlsx")
    out_df.to_excel(out_path, index=False)
    print("wrote", out_path)
