import json
import gzip
import os
import time
from pathlib import Path

import torch
import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
VIDEO_IN = "data/test.mp4"
OUT_DIR = Path("outputs_r1")
BATCH_SIZE = 30

OUT_DIR.mkdir(exist_ok=True)

DONE_FILE = OUT_DIR / "DONE"
START_FILE = OUT_DIR / "START.json"
if DONE_FILE.exists():
    DONE_FILE.unlink()

model = YOLO(WEIGHTS)

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# write START marker (for stable delay)
START_FILE.write_text(
    json.dumps({"video_start_ts": time.time(), "fps": float(fps)}),
    encoding="utf-8",
)
print("Created START marker ->", START_FILE)

stats_path = OUT_DIR / "sizes_and_datarates.txt"

def bytes_to_kb(n_bytes: int) -> float:
    return n_bytes / 1024.0

def bytes_per_sec_to_kbps(bytes_per_sec: float) -> float:
    return (bytes_per_sec * 8.0) / 1000.0

def write_batch(batch_frames, start_frame, end_frame):
    video_stem = Path(VIDEO_IN).stem
    base_name = f"{video_stem}_r1_frames_{start_frame:06d}_to_{end_frame:06d}"
    gz_path = OUT_DIR / f"{base_name}.json.gz"

    payload = {"frames": batch_frames}

    # write gzipped JSON
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    gz_size_b = os.path.getsize(gz_path)

    n_frames = len(batch_frames)
    duration_s = n_frames / float(fps) if fps else 0.0

    gz_bps = (gz_size_b / duration_s) if duration_s > 0 else 0.0
    gz_kb = bytes_to_kb(gz_size_b)
    gz_kbps = bytes_per_sec_to_kbps(gz_bps)

    with open(stats_path, "a", encoding="utf-8") as sf:
        sf.write(
            f"{base_name}\n"
            f"  frames: {n_frames}, duration: {duration_s:.3f}s (fps={fps:.3f})\n"
            f"  gz:   {gz_kb:.2f} KB | rate: {gz_kbps:.2f} kbps\n\n"
        )

    print("Saved:", gz_path)

batch_frames = []
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(frame, conf=0.20, iou=0.5, device=device, verbose=False)[0]

    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            dets.append(
                {
                    "class_id": int(k),
                    "class_name": str(names.get(int(k), int(k))),
                    "conf": float(c),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    batch_frames.append({"frame": frame_idx, "detections": dets})

    if len(batch_frames) == BATCH_SIZE:
        write_batch(batch_frames, batch_frames[0]["frame"], batch_frames[-1]["frame"])
        batch_frames = []

    frame_idx += 1

cap.release()

if batch_frames:
    write_batch(batch_frames, batch_frames[0]["frame"], batch_frames[-1]["frame"])

print("Wrote stats ->", stats_path)

DONE_FILE.write_text("", encoding="utf-8")
print("Created DONE marker ->", DONE_FILE)