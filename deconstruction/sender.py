# sender_watch_outputs_r1.py
#
# Watches outputs_r1/ for new batch files and sends them to your reconstruction PC over WebSocket.
# Supports .json and .json.gz. After successful send, it moves the file to outputs_r1/sent/
#
# pip install websockets

import asyncio
import gzip
import json
import shutil
from pathlib import Path

import websockets

WATCH_DIR = Path("outputs_r1")
SENT_DIR = WATCH_DIR / "sent"
SENT_DIR.mkdir(parents=True, exist_ok=True)

# your reconstruction computer IP
RECEIVER_WS = "ws://127.0.0.1:8765"

POLL_SECONDS = 0.25
MAX_SIZE = 50_000_000  # 50MB

def is_ready_file(p: Path) -> bool:
    # ignore temp/partial files
    return p.is_file() and not p.name.endswith(".tmp") and p.suffix in {".json", ".gz"}

def stable_size(p: Path) -> bool:
    # ensure file is done writing (size stable across two checks)
    s1 = p.stat().st_size
    s2 = p.stat().st_size
    return s1 == s2 and s1 > 0

def load_payload_bytes(p: Path) -> bytes:
    if p.suffix == ".gz":
        # Already gzipped JSON
        return p.read_bytes()
    else:
        # Plain JSON -> gzip it for transport
        raw = p.read_text(encoding="utf-8")
        return gzip.compress(raw.encode("utf-8"), compresslevel=6)

async def send_file(ws, p: Path):
    data = load_payload_bytes(p)
    await ws.send(data)

async def main():
    sent_names = set(x.name for x in SENT_DIR.glob("*"))

    while True:
        try:
            async with websockets.connect(RECEIVER_WS, max_size=MAX_SIZE) as ws:
                print("Connected to", RECEIVER_WS)

                while True:
                    # find unsent files (deterministic order)
                    candidates = sorted(
                        [p for p in WATCH_DIR.iterdir() if is_ready_file(p) and p.name not in sent_names],
                        key=lambda x: x.name,
                    )

                    if not candidates:
                        await asyncio.sleep(POLL_SECONDS)
                        continue

                    for p in candidates:
                        # wait until stable (finished writing)
                        for _ in range(10):
                            if stable_size(p):
                                break
                            await asyncio.sleep(0.1)
                        else:
                            # still not stable, skip for now
                            continue

                        # send
                        await send_file(ws, p)

                        # move to sent/ only after successful send
                        dst = SENT_DIR / p.name
                        shutil.move(str(p), str(dst))
                        sent_names.add(dst.name)

                        print("Sent and moved:", dst.name)

        except Exception as e:
            print("Connection error:", e)
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(main())
