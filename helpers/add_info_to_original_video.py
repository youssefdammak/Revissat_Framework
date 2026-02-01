import cv2
import numpy as np
import time
from pathlib import Path

# ===================== CONFIG =====================

INPUT_VIDEO = "test.mp4"          # <-- your original video
OUTPUT_VIDEO = "output_clock.mp4"  # <-- output video

FONT_SCALE = 1.4
THICKNESS = 3
TEXT_COLOR = (0, 255, 0)           # green
BG_COLOR = (0, 0, 0)               # black
ALPHA = 0.55                       # background transparency

POSITION = (30, 60)                # top-left of text
FOURCC = "mp4v"                    # try "avc1" if needed

# ===================== HELPERS =====================

def format_clock(elapsed_s: float) -> str:
    minutes = int(elapsed_s // 60)
    seconds = int(elapsed_s % 60)
    milliseconds = int((elapsed_s - int(elapsed_s)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def overlay_text_box(
    img,
    text,
    org,
    font_scale,
    thickness,
    text_color,
    bg_color,
    alpha,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = org
    pad = 10

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + pad),
        bg_color,
        -1,
    )

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

# ===================== MAIN =====================

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS ({frame_count} frames)")

    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*FOURCC),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    start_ts = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_s = frame_idx / fps
        clock_text = f"Clock : {format_clock(elapsed_s)}"

        overlay_text_box(
            frame,
            clock_text,
            POSITION,
            FONT_SCALE,
            THICKNESS,
            TEXT_COLOR,
            BG_COLOR,
            ALPHA,
        )

        writer.write(frame)

        cv2.imshow("Clock Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Saved ->", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
