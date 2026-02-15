from pathlib import Path
import json
import time

import cv2
import torch
from ultralytics import YOLO

IN_DIR = Path("batches_15")  # can be batches_30 too; algorithm adapts either way
OUT_DIR = Path("outputs_r1")
OUT_DIR.mkdir(exist_ok=True)

WEIGHTS = "weights/best.pt"
CONF = 0.20
IOU = 0.50

# ---- Controller knobs (simple, paper-friendly) ----
TARGET_S = 0.95   # aim slightly below 1.0s to avoid occasional overruns
ADD = 1           # additive increase when under target
BETA = 0.7        # multiplicative decrease when over target
MIN_KEEP = 1      # min frames processed per batch
# ---------------------------------------------------

model = YOLO(WEIGHTS)
device = 0 if torch.cuda.is_available() else "cpu"

done = set()
keep_n = None  # initialized after seeing first batch size


def pick_evenly(items, n):
    """Pick n items evenly spaced from items (keeps order)."""
    if n >= len(items):
        return items
    step = len(items) / n
    idxs = [int(round(i * step)) for i in range(n)]
    idxs = [min(len(items) - 1, x) for x in idxs]
    # remove duplicates while keeping order
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            out.append(items[i])
            seen.add(i)
    return out


while True:
    for batch_dir in sorted([p for p in IN_DIR.iterdir() if p.is_dir() and p.name.startswith("batch_")]):
        if batch_dir.name in done:
            continue

        frames = sorted(batch_dir.glob("frame_*.jpg"))
        if not frames:
            continue

        if keep_n is None:
            keep_n = len(frames)  # start by processing all frames available in the batch

        # choose subset to process this batch
        frames_to_process = pick_evenly(frames, max(MIN_KEEP, min(keep_n, len(frames))))

        t0 = time.perf_counter()

        out_frames = []
        for img_path in frames_to_process:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            res = model.predict(img, conf=CONF, iou=IOU, device=device, verbose=False)[0]

            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                names = res.names
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    dets.append({
                        "class_id": int(k),
                        "class_name": str(names.get(int(k), int(k))),
                        "conf": float(c),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    })

            frame_num = int(img_path.stem.split("_")[1])
            out_frames.append({"frame": frame_num, "detections": dets})

        out_path = OUT_DIR / f"{batch_dir.name}.json"
        out_path.write_text(json.dumps({"frames": out_frames}), encoding="utf-8")

        t1 = time.perf_counter()
        proc_s = t1 - t0

        # AIMD update for NEXT batch
        if proc_s <= TARGET_S:
            keep_n = min(len(frames), keep_n + ADD)
        else:
            keep_n = max(MIN_KEEP, int(keep_n * BETA))

        print(
            f"{batch_dir.name} processed in {proc_s:.3f}s "
            f"(kept {len(frames_to_process)}/{len(frames)} frames) -> {out_path.name} | next_keep={keep_n}"
        )

        done.add(batch_dir.name)

    time.sleep(0.1)