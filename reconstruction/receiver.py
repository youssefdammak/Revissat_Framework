# receiver.py
# pip install websockets opencv-python numpy

import asyncio
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

PORT = 8765

BACKGROUND_PATH = "background.jpg"  # set None for black background
OUT_VIDEO = "reconstructed.mp4"
FOURCC = "mp4v"  # try "avc1" if mp4v fails

INITIAL_BUFFER_FRAMES = 150
MAX_STORE_FRAMES = 20000

# ===================== STATE =====================

detections_by_frame = {}
max_received_frame = -1
eos_received = False
lock = asyncio.Lock()

fps = None
batch_ranges = []  # {batch_id,start_frame,end_frame,kbps}

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


def format_clock(elapsed_s: float) -> str:
    minutes = int(elapsed_s // 60)
    seconds = int(elapsed_s % 60)
    milliseconds = int((elapsed_s - int(elapsed_s)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def render_frame(bg, frame_idx: int, dets, clock_s: float, delay_s: float):
    frame = bg.copy()
    if dets:
        draw_detections(frame, dets)

    br = find_batch_for_frame(frame_idx)
    bid = br["batch_id"] if br else -1
    kbps = br["kbps"] if br else 0.0

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
    global max_received_frame, eos_received, fps

    async for msg in ws:
        if isinstance(msg, str):
            data = json.loads(msg)
            t = data.get("type")
            if t == "init":
                try:
                    fps = float(data.get("fps", fps if fps is not None else 30.0))
                except Exception:
                    fps = fps if fps is not None else 30.0
                print("INIT:", {"fps": fps})
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

        if fps is None:
            try:
                fps = float(data.get("fps", 30.0))
            except Exception:
                fps = 30.0

        frames = data.get("frames", [])
        duration_s = (len(frames) / fps) if fps and fps > 0 else 0.0
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
    bg = load_background()
    if bg is None:
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)

    H, W = bg.shape[:2]
    Path(OUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Reconstruction", cv2.WINDOW_NORMAL)

    # Wait until we have fps + at least one received frame (or EOS)
    while True:
        async with lock:
            mr = max_received_frame
            eos = eos_received
        if (fps is not None and fps > 0 and mr >= 0) or eos:
            break
        await asyncio.sleep(0.02)

    out_fps = float(fps) if fps and fps > 0 else 30.0
    frame_step = 1.0 / max(out_fps, 1e-6)

    writer = cv2.VideoWriter(
        OUT_VIDEO,
        cv2.VideoWriter_fourcc(*FOURCC),
        out_fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    # Choose where to start showing (buffer)
    async with lock:
        mr = max_received_frame
    play_frame = max(0, mr - INITIAL_BUFFER_FRAMES)

    # Real timer: starts at 0 and keeps increasing (monotonic)
    clock_start = time.perf_counter()

    # Drive output at fps, writing every tick.
    next_tick = time.perf_counter()

    while True:
        now = time.perf_counter()
        sleep_s = next_tick - now
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        next_tick += frame_step

        async with lock:
            mr = max_received_frame
            eos = eos_received

        # Clock always moves
        clock_s = time.perf_counter() - clock_start

        # If we have not received up to play_frame yet, clamp to last received
        # BUT still write frames (duplicates) so clock keeps moving in the saved video.
        if mr >= 0 and play_frame > mr:
            show_frame = mr
        else:
            show_frame = play_frame

        async with lock:
            dets = detections_by_frame.get(show_frame, [])

        # Original time is where we are in the original timeline (based on play_frame progress)
        original_t = (play_frame / out_fps) if out_fps > 0 else 0.0

        # Generated time is the real clock
        generated_t = clock_s

        # Your requested metric:
        delay_s = generated_t - original_t

        frame = render_frame(bg, show_frame, dets, generated_t, delay_s)
        writer.write(frame)
        cv2.imshow("Reconstruction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Advance original timeline ONLY if the next frame exists
        # (this is what makes delay grow when network/processing is behind)
        async with lock:
            mr2 = max_received_frame
            eos2 = eos_received

        if play_frame <= mr2:
            play_frame += 1

        # End condition: EOS and we've advanced past the last received frame
        if eos2 and mr2 >= 0 and play_frame > mr2:
            break

    writer.release()
    cv2.destroyAllWindows()
    print("Saved ->", OUT_VIDEO)


# ===================== MAIN =====================

async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=50_000_000):
        print(f"Listening on ws://127.0.0.1:{PORT}")
        await render_and_save()

asyncio.run(main())