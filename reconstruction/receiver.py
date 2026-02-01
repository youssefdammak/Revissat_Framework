# reconstruction/receiver_ws_save_video.py
# pip install websockets opencv-python numpy

import asyncio
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

# ===================== CONFIG =====================

PORT = 8765

BACKGROUND_PATH = "background.jpg"   # set None for black background
OUT_VIDEO = "reconstructed.mp4"
FOURCC = "mp4v"                      # try "avc1" if mp4v fails

INITIAL_BUFFER_FRAMES = 150
MAX_STORE_FRAMES = 20000
PREROLL_PREVIEW_EVERY = 10

# ===================== STATE =====================

detections_by_frame = {}
max_received_frame = -1
eos_received = False
lock = asyncio.Lock()

video_start_ts = None        # sender capture start (epoch)
video_play_start_ts = None   # local playback start
fps = 30.0

batch_ranges = []  # list of {batch_id,start_frame,end_frame,kbps}

# ===================== HELPERS =====================

def load_background():
    if BACKGROUND_PATH is None:
        return None
    bg = cv2.imread(BACKGROUND_PATH)
    if bg is None:
        raise RuntimeError(f"Failed to read background image: {BACKGROUND_PATH}")
    return bg


def draw_detections(img, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d.get("class_name","?")} {d.get("conf",0):.2f}'
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def overlay_text_box(
    img,
    lines,
    org=(30, 60),
    font_scale=1.4,
    thickness=3,
    text_color=(0, 255, 0),
    bg_color=(0, 0, 0),
    alpha=0.55,
    line_spacing=10,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]

    width = max(w for w, h in sizes) + 20
    height = sum(h for w, h in sizes) + line_spacing * (len(lines) - 1) + 20

    x, y = org
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x - 10, y - sizes[0][1] - 10),
        (x - 10 + width, y - sizes[0][1] - 10 + height),
        bg_color,
        -1,
    )

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    yy = y
    for (line, (w, h)) in zip(lines, sizes):
        cv2.putText(img, line, (x, yy), font, font_scale, text_color, thickness)
        yy += h + line_spacing


def find_batch_for_frame(frame_idx: int):
    for br in batch_ranges:
        if br["start_frame"] <= frame_idx <= br["end_frame"]:
            return br
    return None


def compute_delay(play_frame: int) -> float:
    if video_start_ts is None or fps <= 0:
        return 0.0
    capture_ts = video_start_ts + (play_frame / fps)
    return max(0.0, time.time() - capture_ts)


def format_clock(elapsed_s: float) -> str:
    minutes = int(elapsed_s // 60)
    seconds = int(elapsed_s % 60)
    milliseconds = int((elapsed_s - int(elapsed_s)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def render_frame(bg, frame_idx: int, dets):
    frame = bg.copy()
    if dets:
        draw_detections(frame, dets)

    br = find_batch_for_frame(frame_idx)
    bid = br["batch_id"] if br else -1
    kbps = br["kbps"] if br else 0.0
    delay_s = compute_delay(frame_idx)

    clock_s = 0.0
    if video_play_start_ts is not None:
        clock_s = time.time() - video_play_start_ts

    overlay_text_box(
        frame,
        [
            f"Batch {bid} : {kbps:.1f} kbps",
            f"Delay : {delay_s:.2f} s",
            f"Clock : {format_clock(clock_s)}",
        ],
    )

    return frame

# ===================== WEBSOCKET HANDLER =====================

async def handler(ws):
    global max_received_frame, eos_received, video_start_ts, fps

    async for msg in ws:
        if isinstance(msg, str):
            data = json.loads(msg)
            t = data.get("type")
            if t == "init":
                v = float(data.get("video_start_ts", 0.0))
                if v >= 1_000_000_000:
                    video_start_ts = v
                fps = float(data.get("fps", fps))
                print("INIT:", {"video_start_ts": video_start_ts, "fps": fps})
            elif t == "eos":
                eos_received = True
                print("EOS received")
            continue

        if not isinstance(msg, (bytes, bytearray)):
            continue

        wire_bytes = len(msg)
        data = json.loads(gzip.decompress(msg).decode("utf-8"))

        if data.get("type") != "r1_batch":
            continue

        batch_id = int(data["batch_id"])
        start_f = int(data["start_frame"])
        end_f = int(data["end_frame"])

        frames = data.get("frames", [])
        duration_s = (len(frames) / fps) if fps > 0 else 0.0
        kbps = ((wire_bytes * 8.0) / (duration_s * 1000.0)) if duration_s > 0 else 0.0

        async with lock:
            for f in frames:
                fi = int(f["frame"])
                detections_by_frame[fi] = f.get("detections", [])
                max_received_frame = max(max_received_frame, fi)

            batch_ranges.append(
                {"batch_id": batch_id, "start_frame": start_f, "end_frame": end_f, "kbps": kbps}
            )

            if len(detections_by_frame) > MAX_STORE_FRAMES:
                for k in sorted(detections_by_frame)[:-MAX_STORE_FRAMES]:
                    detections_by_frame.pop(k, None)

        print(f"Got batch{batch_id} {start_f}->{end_f} kbps={kbps:.1f}")

# ===================== RENDER LOOP =====================

async def render_and_save():
    global video_play_start_ts

    bg = load_background()
    if bg is None:
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)

    H, W = bg.shape[:2]
    Path(OUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        OUT_VIDEO,
        cv2.VideoWriter_fourcc(*FOURCC),
        fps,
        (W, H),
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    cv2.namedWindow("Reconstruction", cv2.WINDOW_NORMAL)

    while True:
        async with lock:
            mr = max_received_frame
            eos = eos_received
        if mr >= 0 or eos:
            break
        await asyncio.sleep(0.02)

    video_play_start_ts = time.time()

    preroll_end = max(0, mr - INITIAL_BUFFER_FRAMES)
    for fidx in range(preroll_end):
        async with lock:
            dets = detections_by_frame.get(fidx, [])
        frame = render_frame(bg, fidx, dets)
        writer.write(frame)

    play_frame = preroll_end
    frame_time = 1.0 / max(fps, 1e-6)
    last = time.perf_counter()

    while True:
        dt = time.perf_counter() - last
        if dt < frame_time:
            await asyncio.sleep(frame_time - dt)
        last = time.perf_counter()

        async with lock:
            dets = detections_by_frame.get(play_frame, [])
            mr = max_received_frame
            eos = eos_received

        frame = render_frame(bg, play_frame, dets)
        writer.write(frame)
        cv2.imshow("Reconstruction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        play_frame += 1
        if eos and play_frame > mr:
            break
        if play_frame > mr and not eos:
            play_frame = mr

    writer.release()
    cv2.destroyAllWindows()
    print("Saved ->", OUT_VIDEO)

# ===================== MAIN =====================

async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=50_000_000):
        print(f"Listening on ws://127.0.0.1:{PORT}")
        await render_and_save()

asyncio.run(main())
