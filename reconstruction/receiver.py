# receiver_ws.py
# pip install websockets

import asyncio
import gzip
import json
import websockets

async def handler(ws):
    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            data = json.loads(gzip.decompress(msg).decode("utf-8"))
        else:
            data = json.loads(msg)

        # data["frames"] is your R1 batch
        print("got batch frames:", len(data.get("frames", [])))

        # TODO: recon.update_from_r1_batch(data)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=50_000_000):
        print("Listening on ws://0.0.0.0:8765")
        await asyncio.Future()

asyncio.run(main())
