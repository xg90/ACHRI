import csv
import math
from pathlib import Path
from openpyxl import Workbook

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / 'data' / 'openface'
BATCHES = [BASE_DIR / 'batch1', BASE_DIR / 'batch2']

AU_INTENSITY = [
    'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r',
    'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r'
]
AU_PRESENCE = [
    'AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c',
    'AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c'
]
OTHER = ['gaze_angle_x','gaze_angle_y','pose_Rx','pose_Ry','pose_Rz']
FEATURES = AU_INTENSITY + AU_PRESENCE + OTHER
OUTPUT_HEADER = ['video_id'] + FEATURES

CHUNK_SIZE = 320
NUM_CHUNKS = 6
MAX_FRAMES = CHUNK_SIZE * NUM_CHUNKS


def safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def compute_baseline_mean(baseline_path: Path, feature_cols):
    sums = {c: 0.0 for c in feature_cols}
    counts = {c: 0 for c in feature_cols}
    with baseline_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in feature_cols:
                v = safe_float(row.get(c, ''))
                if v is None:
                    continue
                sums[c] += v
                counts[c] += 1
    means = {}
    for c in feature_cols:
        if counts[c] == 0:
            means[c] = 0.0
        else:
            means[c] = sums[c] / counts[c]
    return means


def load_corrected_rows(obs_path: Path, feature_cols, baseline_means):
    rows = []
    with obs_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        # validate columns
        header = reader.fieldnames or []
        missing = [c for c in ['frame'] + feature_cols if c not in header]
        if missing:
            raise ValueError(f'Missing columns in {obs_path}: {missing}')
        for i, row in enumerate(reader):
            if i >= MAX_FRAMES:
                break
            frame_v = safe_float(row.get('frame', ''))
            _ = frame_v  # keep parse for potential validation if needed later
            out = [None]  # placeholder for video_id
            for c in feature_cols:
                v = safe_float(row.get(c, ''))
                v = v if v is not None else 0.0
                out.append(v - baseline_means[c])
            rows.append(out)
    return rows


def chunk_mean(rows):
    if not rows:
        return [None] + [0.0] * (len(OUTPUT_HEADER) - 1)
    n = len(rows)
    acc = [0.0] * (len(rows[0]) - 1)
    for r in rows:
        for i, v in enumerate(r[1:]):
            acc[i] += v
    return [None] + [v / n for v in acc]


def process_video(video_dir: Path):
    baseline_path = video_dir / 'baseline.csv'
    obs_path = video_dir / 'observation.csv'
    if not baseline_path.exists() or not obs_path.exists():
        raise FileNotFoundError(f'Missing baseline/observation in {video_dir}')
    baseline_means = compute_baseline_mean(baseline_path, FEATURES)
    rows = load_corrected_rows(obs_path, FEATURES, baseline_means)
    chunks = [rows[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(NUM_CHUNKS)]
    return [chunk_mean(c) for c in chunks]


def process_batch(batch_dir: Path, output_name: str):
    video_dirs = [p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith('video_')]
    def sort_key(p: Path):
        try:
            return int(p.name.split('_')[-1])
        except Exception:
            return p.name
    video_dirs.sort(key=sort_key)

    all_rows = []
    for vd in video_dirs:
        video_id = vd.name
        chunk_rows = process_video(vd)
        for r in chunk_rows:
            r[0] = video_id
        all_rows.extend(chunk_rows)

    wb = Workbook()
    ws = wb.active
    ws.title = 'data'
    ws.append(OUTPUT_HEADER)
    for r in all_rows:
        ws.append(r)

    out_path = batch_dir / output_name
    wb.save(out_path)
    return out_path, len(all_rows)


def main():
    for batch_dir, out_name in [(BATCHES[0], 'OPENFACE1.xlsx'), (BATCHES[1], 'OPENFACE2.xlsx')]:
        out_path, nrows = process_batch(batch_dir, out_name)
        print(f'Wrote {nrows} rows to {out_path}')


if __name__ == '__main__':
    main()
