import json
import gzip
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
VIDEO_IN = "data/test.mp4"
OUT_DIR = Path("outputs_r3")
BATCH_SIZE = 30
CONF = 0.20
IOU = 0.50

OUT_DIR.mkdir(exist_ok=True)

model = YOLO(WEIGHTS)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

stats_path = OUT_DIR / "sizes_and_datarates.txt"

def bytes_to_kb(n_bytes: int) -> float:
    return n_bytes / 1024.0

def bytes_per_sec_to_kbps(bps: float) -> float:
    return (bps * 8.0) / 1000.0

def write_outputs(window_name: str, window_dir: Path, payload: dict, frames_in_batch: int):
    json_path = window_dir / f"r3_{window_name}.json"
    gz_path = window_dir / f"r3_{window_name}.json.gz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    json_b = os.path.getsize(json_path)
    gz_b = os.path.getsize(gz_path)

    duration_s = frames_in_batch / float(fps) if fps else 0.0
    json_kbps = bytes_per_sec_to_kbps(json_b / duration_s) if duration_s > 0 else 0.0
    gz_kbps = bytes_per_sec_to_kbps(gz_b / duration_s) if duration_s > 0 else 0.0

    with open(stats_path, "a", encoding="utf-8") as sf:
        sf.write(
            f"{window_name}\n"
            f"  frames: {frames_in_batch}, duration: {duration_s:.3f}s (fps={fps:.3f})\n"
            f"  json: {bytes_to_kb(json_b):.2f} KB | rate: {json_kbps:.2f} kbps\n"
            f"  gz:   {bytes_to_kb(gz_b):.2f} KB | rate: {gz_kbps:.2f} kbps\n\n"
        )

    print("Saved:", json_path)
    print("Saved:", gz_path)

frame_idx = 0  # 0-based like R1
batch_idx = 0

while True:
    start_frame = batch_idx * BATCH_SIZE
    end_frame = start_frame + (BATCH_SIZE - 1)
    window_name = f"frame_{start_frame}_to_{end_frame}"
    window_dir = OUT_DIR / window_name
    window_dir.mkdir(exist_ok=True)

    frames_in_batch = 0
    batch_frames = []   # same structure as R1: list of {"frame": idx, "detections": [...]}
    had_detection = False
    last_frame_img = None
    last_frame_idx = None

    for _ in range(BATCH_SIZE):
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=CONF, iou=IOU, verbose=False)[0]

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

        if dets:
            had_detection = True

        batch_frames.append({"frame": frame_idx, "detections": dets})

        last_frame_img = frame
        last_frame_idx = frame_idx

        frame_idx += 1
        frames_in_batch += 1

    if frames_in_batch == 0:
        break

    # If batch has any detection, save last frame image
    if had_detection and last_frame_img is not None:
        img_name = f"r3_{window_name}_last_frame.jpg"
        img_path = window_dir / img_name
        cv2.imwrite(str(img_path), last_frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        payload = {
            "frames": batch_frames,
            "last_frame": {
                "frame": int(last_frame_idx),
                "image": img_name,
            },
        }

        write_outputs(window_name, window_dir, payload, frames_in_batch)
    else:
        # No detection: leave folder empty (no json, no image)
        pass

    batch_idx += 1

cap.release()

print("Wrote stats ->", stats_path)
print("Done. Output dir:", OUT_DIR)
