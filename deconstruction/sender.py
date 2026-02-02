# sender.py
import asyncio
import gzip
import json
import re
import shutil
import time
from pathlib import Path

import websockets

WATCH_DIR = Path("outputs_r1")
SENT_DIR = WATCH_DIR / "sent"
DONE_FILE = WATCH_DIR / "DONE"
START_FILE = WATCH_DIR / "START.json"

SENT_DIR.mkdir(parents=True, exist_ok=True)

RECEIVER_WS = "ws://127.0.0.1:8765"
POLL_SECONDS = 0.25
MAX_SIZE = 50_000_000

# Match: *_r1_frames_000000_to_000029.json.gz
RANGE_RE = re.compile(r"_frames_(\d+)_to_(\d+)\.json\.gz$")

BATCH_SIZE = 30  # must match exporter


def stable_size(p: Path) -> bool:
    s1 = p.stat().st_size
    s2 = p.stat().st_size
    return s1 == s2 and s1 > 0


def read_start_info():
    data = json.loads(START_FILE.read_text(encoding="utf-8"))
    return float(data["video_start_ts"]), float(data["fps"])


def parse_range_from_name(name: str):
    m = RANGE_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def cleanup_plain_json(sent_names: set[str]):
    for p in WATCH_DIR.glob("*.json"):
        if p.name == "START.json":
            continue
        if p.name in sent_names:
            continue
        dst = SENT_DIR / p.name
        if dst.exists():
            sent_names.add(p.name)
            try:
                p.unlink()
            except Exception:
                pass
            continue
        try:
            shutil.move(str(p), str(dst))
            sent_names.add(dst.name)
        except Exception:
            pass


def load_frames_from_gz(p: Path) -> list:
    payload = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
    return payload.get("frames", [])


async def main():
    sent_names = set(x.name for x in SENT_DIR.glob("*"))

    video_start_ts = None
    fps = 30.0

    for _ in range(120):  # up to ~30s
        if START_FILE.exists():
            try:
                video_start_ts, fps = read_start_info()
                break
            except Exception:
                pass
        await asyncio.sleep(0.25)

    if video_start_ts is None:
        video_start_ts = time.time()
        fps = 30.0
        print(
            "WARNING: START.json missing/unreadable. Using fallback:",
            {"video_start_ts": video_start_ts, "fps": fps},
        )
    else:
        print("Loaded START.json:", {"video_start_ts": video_start_ts, "fps": fps})

    while True:
        try:
            async with websockets.connect(RECEIVER_WS, max_size=MAX_SIZE) as ws:
                print("Connected to", RECEIVER_WS)

                await ws.send(
                    json.dumps(
                        {
                            "type": "init",
                            "video_start_ts": float(video_start_ts),
                            "fps": float(fps),
                        }
                    )
                )

                while True:
                    cleanup_plain_json(sent_names)

                    candidates = sorted(
                        [
                            p
                            for p in WATCH_DIR.glob("*.json.gz")
                            if p.is_file() and p.name not in sent_names
                        ],
                        key=lambda x: x.name,
                    )

                    if not candidates:
                        if DONE_FILE.exists():
                            await ws.send(json.dumps({"type": "eos"}))
                            print("Sent EOS (DONE found, no pending .json.gz). Exiting sender.")
                            return
                        await asyncio.sleep(POLL_SECONDS)
                        continue

                    for p in candidates:
                        for _ in range(30):
                            if stable_size(p):
                                break
                            await asyncio.sleep(0.1)
                        else:
                            continue

                        name_range = parse_range_from_name(p.name)
                        frames = load_frames_from_gz(p)

                        if name_range is not None:
                            start_f, end_f = name_range
                        else:
                            if frames:
                                start_f = int(frames[0]["frame"])
                                end_f = int(frames[-1]["frame"])
                            else:
                                start_f, end_f = -1, -1

                        batch_id = (start_f // BATCH_SIZE) if start_f >= 0 else -1

                        wrapped = {
                            "type": "r1_batch",
                            "batch_id": int(batch_id),
                            "start_frame": int(start_f),
                            "end_frame": int(end_f),
                            "fps": float(fps),
                            "video_start_ts": float(video_start_ts),
                            "frames": frames,
                        }

                        out_bytes = gzip.compress(
                            json.dumps(wrapped, separators=(",", ":")).encode("utf-8"),
                            compresslevel=6,
                        )
                        await ws.send(out_bytes)

                        dst = SENT_DIR / p.name
                        shutil.move(str(p), str(dst))
                        sent_names.add(dst.name)

                        print(f"Sent batch{batch_id} {start_f}->{end_f} moved:", dst.name)

        except Exception as e:
            print("Connection error:", e)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main())