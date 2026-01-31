import json
import gzip
import shutil
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# Paths
WEIGHTS = "weights/best.pt"
VIDEO_IN = "data/test.mp4"
OUT_DIR = Path("outputs_r2")
PATCHES_DIR = OUT_DIR / "patches"
BATCH_SIZE = 30
CONF = 0.20
IOU = 0.50

# Patch compression knobs
JPEG_QUALITY = 55
GRAYSCALE = False

OUT_DIR.mkdir(exist_ok=True)
PATCHES_DIR.mkdir(exist_ok=True)

model = YOLO(WEIGHTS)
names = model.names

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

def clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))

def safe_crop(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]
    x1i = clamp(x1, 0, W - 1)
    y1i = clamp(y1, 0, H - 1)
    x2i = clamp(x2, 0, W)
    y2i = clamp(y2, 0, H)
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = frame[y1i:y2i, x1i:x2i]
    return crop if crop.size else None

def encode_jpeg(img) -> bytes:
    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
    ok, buf = cv2.imencode(".jpg", img, params)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def write_window_outputs(window_name: str, window_dir: Path, patches_meta: dict, selected_global_paths: dict):
    window_dir.mkdir(exist_ok=True)

    # Copy selected patches into window folder and update JSON patch paths to local filenames
    for class_name, src_abs in selected_global_paths.items():
        if not src_abs:
            continue
        src_abs = Path(src_abs)
        if not src_abs.exists():
            continue
        dst_name = src_abs.name
        dst_abs = window_dir / dst_name
        shutil.copy2(src_abs, dst_abs)
        patches_meta[class_name]["patch"] = dst_name  # local to window folder

    payload = {"patches": {window_name: patches_meta}}

    json_path = window_dir / f"r2_{window_name}.json"
    gz_path = window_dir / f"r2_{window_name}.json.gz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    json_size_b = os.path.getsize(json_path)
    gz_size_b = os.path.getsize(gz_path)

    patch_total_b = 0
    for v in patches_meta.values():
        p = window_dir / v["patch"]
        if p.exists():
            patch_total_b += p.stat().st_size

    return json_size_b, gz_size_b, patch_total_b

# --- Main loop: pick HIGHEST confidence per class within each batch ---
frame_idx = 0
current_window = None
frames_in_current_window = 0

patches_for_window = {}     # {class_name: meta}
selected_patch_abs = {}     # {class_name: abs path}
best_conf_for_window = {}   # {class_name: best_conf}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    window_start = (frame_idx // BATCH_SIZE) * BATCH_SIZE + 1
    window_end = window_start + (BATCH_SIZE - 1)
    window_name = f"frame_{window_start}_to_{window_end}"

    # window change -> flush previous
    if current_window is None:
        current_window = window_name
        frames_in_current_window = 0
    elif window_name != current_window:
        window_dir = OUT_DIR / current_window

        json_b, gz_b, patches_b = write_window_outputs(current_window, window_dir, patches_for_window, selected_patch_abs)

        duration_s = frames_in_current_window / float(fps) if fps else 0.0
        json_kbps = bytes_per_sec_to_kbps(json_b / duration_s) if duration_s > 0 else 0.0
        gz_kbps = bytes_per_sec_to_kbps(gz_b / duration_s) if duration_s > 0 else 0.0
        patches_kbps = bytes_per_sec_to_kbps(patches_b / duration_s) if duration_s > 0 else 0.0

        with open(stats_path, "a", encoding="utf-8") as sf:
            sf.write(
                f"{current_window}\n"
                f"  frames: {frames_in_current_window}, duration: {duration_s:.3f}s (fps={fps:.3f})\n"
                f"  json: {bytes_to_kb(json_b):.2f} KB | rate: {json_kbps:.2f} kbps\n"
                f"  gz:   {bytes_to_kb(gz_b):.2f} KB | rate: {gz_kbps:.2f} kbps\n"
                f"  patches (referenced): {bytes_to_kb(patches_b):.2f} KB | rate: {patches_kbps:.2f} kbps\n\n"
            )

        # reset
        current_window = window_name
        frames_in_current_window = 0
        patches_for_window = {}
        selected_patch_abs = {}
        best_conf_for_window = {}

    # detect
    res = model.predict(frame, conf=CONF, iou=IOU, verbose=False)[0]

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            c = float(c)
            class_name = str(names.get(int(k), int(k)))

            # only update if this detection is better than current best in this window
            if class_name in best_conf_for_window and c <= best_conf_for_window[class_name]:
                continue

            crop = safe_crop(frame, x1, y1, x2, y2)
            if crop is None:
                continue

            jpg_bytes = encode_jpeg(crop)

            patch_filename = f"{current_window}_{class_name}_{frame_idx:06d}.jpg"
            patch_path = PATCHES_DIR / patch_filename
            with open(patch_path, "wb") as pf:
                pf.write(jpg_bytes)

            best_conf_for_window[class_name] = c
            patches_for_window[class_name] = {
                "patch": patch_filename,  # will become local name on flush
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": c,
                "encoding": {
                    "format": "jpeg",
                    "quality": JPEG_QUALITY,
                    "grayscale": GRAYSCALE,
                },
            }
            selected_patch_abs[class_name] = str(patch_path)

    frame_idx += 1
    frames_in_current_window += 1

cap.release()

# flush last window
if current_window is not None:
    window_dir = OUT_DIR / current_window
    json_b, gz_b, patches_b = write_window_outputs(current_window, window_dir, patches_for_window, selected_patch_abs)

    duration_s = frames_in_current_window / float(fps) if fps else 0.0
    json_kbps = bytes_per_sec_to_kbps(json_b / duration_s) if duration_s > 0 else 0.0
    gz_kbps = bytes_per_sec_to_kbps(gz_b / duration_s) if duration_s > 0 else 0.0
    patches_kbps = bytes_per_sec_to_kbps(patches_b / duration_s) if duration_s > 0 else 0.0

    with open(stats_path, "a", encoding="utf-8") as sf:
        sf.write(
            f"{current_window}\n"
            f"  frames: {frames_in_current_window}, duration: {duration_s:.3f}s (fps={fps:.3f})\n"
            f"  json: {bytes_to_kb(json_b):.2f} KB | rate: {json_kbps:.2f} kbps\n"
            f"  gz:   {bytes_to_kb(gz_b):.2f} KB | rate: {gz_kbps:.2f} kbps\n"
            f"  patches (referenced): {bytes_to_kb(patches_b):.2f} KB | rate: {patches_kbps:.2f} kbps\n\n"
        )

print("Wrote stats ->", stats_path)
print("Done. Output dir:", OUT_DIR)
