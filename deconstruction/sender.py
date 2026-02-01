# deconstruction/sender_watch_outputs_r1.py
# pip install websockets

import asyncio
import gzip
import json
import shutil
from pathlib import Path

import websockets

WATCH_DIR = Path("outputs_r1")
SENT_DIR = WATCH_DIR / "sent"
DONE_FILE = WATCH_DIR / "DONE"

SENT_DIR.mkdir(parents=True, exist_ok=True)

RECEIVER_WS = "ws://127.0.0.1:8765"  # same machine
POLL_SECONDS = 0.25
MAX_SIZE = 50_000_000  # 50MB


def is_ready_file(p: Path) -> bool:
    return p.is_file() and not p.name.endswith(".tmp") and p.suffix in {".json", ".gz"}


def stable_size(p: Path) -> bool:
    s1 = p.stat().st_size
    s2 = p.stat().st_size
    return s1 == s2 and s1 > 0


def load_payload_bytes(p: Path) -> bytes:
    if p.suffix == ".gz":
        return p.read_bytes()
    raw = p.read_text(encoding="utf-8")
    return gzip.compress(raw.encode("utf-8"), compresslevel=6)


async def main():
    sent_names = set(x.name for x in SENT_DIR.glob("*"))

    while True:
        try:
            async with websockets.connect(RECEIVER_WS, max_size=MAX_SIZE) as ws:
                print("Connected to", RECEIVER_WS)

                while True:
                    candidates = sorted(
                        [p for p in WATCH_DIR.iterdir() if is_ready_file(p) and p.name not in sent_names],
                        key=lambda x: x.name,
                    )

                    if not candidates:
                        if DONE_FILE.exists():
                            await ws.send(json.dumps({"type": "eos"}))
                            print("Sent EOS (DONE found). Exiting sender.")
                            return
                        await asyncio.sleep(POLL_SECONDS)
                        continue

                    for p in candidates:
                        for _ in range(10):
                            if stable_size(p):
                                break
                            await asyncio.sleep(0.1)
                        else:
                            continue

                        await ws.send(load_payload_bytes(p))

                        dst = SENT_DIR / p.name
                        shutil.move(str(p), str(dst))
                        sent_names.add(dst.name)
                        print("Sent and moved:", dst.name)

        except Exception as e:
            print("Connection error:", e)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main())