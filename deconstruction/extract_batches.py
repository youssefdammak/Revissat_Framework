# script1_extract_batches.py
# Reads video and saves 15 frames per batch by taking every other frame:
# take, skip, take, skip ... (from 30 input frames)
# Output per batch: frame_000000.jpg, frame_000002.jpg, ... (15 files)

from pathlib import Path
import cv2

VIDEO_IN = "data/test.mp4"
OUT_DIR = Path("batches_6")
OUT_DIR.mkdir(exist_ok=True)

INPUT_BATCH = 30          # how many frames define a batch window
TAKE_EVERY = 5            # take 1, skip 1
TAKEN_PER_BATCH = INPUT_BATCH // TAKE_EVERY  # 15

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError("failed to open video")

batch_idx = 0
frame_idx = 0
pos_in_batch = 0

batch_dir = OUT_DIR / f"batch_{batch_idx:06d}"
batch_dir.mkdir(parents=True, exist_ok=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # take frames 0,2,4,... within each 30-frame batch window
    if (pos_in_batch % TAKE_EVERY) == 0:
        cv2.imwrite(str(batch_dir / f"frame_{frame_idx:06d}.jpg"), frame)

    frame_idx += 1
    pos_in_batch += 1

    # move to next batch after 30 input frames, even though we only saved 15
    if pos_in_batch == INPUT_BATCH:
        batch_idx += 1
        pos_in_batch = 0
        batch_dir = OUT_DIR / f"batch_{batch_idx:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

cap.release()
print("done")