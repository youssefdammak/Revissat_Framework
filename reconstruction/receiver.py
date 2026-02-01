# reconstruction/receiver_ws_save_video.py
# pip install websockets opencv-python

import asyncio
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

PORT = 8765

# Set these:
FPS = 30.0
BACKGROUND_PATH = "background.jpg"   # set None for black background
OUT_VIDEO = "reconstructed.mp4"

# Playback buffering (delay is allowed)
INITIAL_BUFFER_FRAMES = 60
MAX_STORE_FRAMES = 5000
FOURCC = "mp4v"

detections_by_frame = {}     # frame_idx -> list[dets]
max_received_frame = -1
eos_received = False
lock = asyncio.Lock()


def load_background():
    if BACKGROUND_PATH is None:
        return None
    bg = cv2.imread(BACKGROUND_PATH)
    if bg is None:
        raise RuntimeError(f"Failed to read background image: {BACKGROUND_PATH}")
    return bg


def draw_detections(img, dets):
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d.get("class_name","?")} {d.get("conf",0):.2f}'
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


async def handler(ws):
    global max_received_frame, eos_received

    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            data = json.loads(gzip.decompress(msg).decode("utf-8"))
        else:
            data = json.loads(msg)

        if data.get("type") == "eos":
            eos_received = True
            print("EOS received")
            continue

        frames = data.get("frames", [])
        async with lock:
            for f in frames:
                fi = int(f["frame"])
                dets = f.get("detections", []) or []
                detections_by_frame[fi] = dets
                if fi > max_received_frame:
                    max_received_frame = fi

            # prune
            if len(detections_by_frame) > MAX_STORE_FRAMES:
                keys = sorted(detections_by_frame.keys())
                for k in keys[: len(keys) - MAX_STORE_FRAMES]:
                    detections_by_frame.pop(k, None)

        print("received batch frames:", len(frames), "max_frame:", max_received_frame)


async def render_and_save():
    bg = load_background()
    if bg is None:
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)

    H, W = bg.shape[:2]

    Path(OUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {OUT_VIDEO} (try FOURCC='avc1' or output .avi)")

    frame_time = 1.0 / FPS

    # wait for buffer or eos (short videos)
    while True:
        async with lock:
            mr = max_received_frame
            eos = eos_received
        if mr >= INITIAL_BUFFER_FRAMES or eos:
            break
        await asyncio.sleep(0.05)

    play_frame = 0
    last_tick = time.perf_counter()

    while True:
        # pacing
        now = time.perf_counter()
        dt = now - last_tick
        if dt < frame_time:
            await asyncio.sleep(frame_time - dt)
        last_tick = time.perf_counter()

        async with lock:
            dets = detections_by_frame.get(play_frame, [])
            mr = max_received_frame
            eos = eos_received

        frame = bg.copy()
        if dets:
            draw_detections(frame, dets)

        writer.write(frame)
        cv2.imshow("Reconstruction (R1 overlay + saving)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested")
            break

        play_frame += 1

        # stop when EOS arrived and we've rendered past last received frame
        if eos and play_frame > mr:
            break

        # if we are ahead and eos not arrived yet, hold on last frame
        if play_frame > mr and not eos:
            play_frame = mr

    writer.release()
    cv2.destroyAllWindows()
    print("Saved video ->", OUT_VIDEO)


async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=50_000_000):
        print(f"Listening on ws://127.0.0.1:{PORT}")
        await render_and_save()

asyncio.run(main())