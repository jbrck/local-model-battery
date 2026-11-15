"""Forward a Docker-bridge address to a loopback-only model server.

llama.cpp/openai endpoints that bind 127.0.0.1 are invisible to sandbox containers, which
reach the host at the docker bridge gateway (172.17.0.1 / host.docker.internal). This
listens on the bridge address and forwards to loopback, so a sandboxed leg can call the
engine without restarting it on 0.0.0.0.

    python3 endpoint_bridge.py --listen 172.17.0.1:8124 --target 127.0.0.1:8124
"""
from __future__ import annotations

import argparse
import asyncio
import sys


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def _handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter,
                  target_host: str, target_port: int) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
    except OSError as error:
        print(f"bridge: cannot reach {target_host}:{target_port}: {error}", file=sys.stderr)
        client_writer.close()
        return
    await asyncio.gather(_pump(client_reader, upstream_writer),
                         _pump(upstream_reader, client_writer))


async def serve(listen_host: str, listen_port: int, target_host: str, target_port: int) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, target_host, target_port), listen_host, listen_port,
        reuse_address=True)
    print(f"bridge: {listen_host}:{listen_port} -> {target_host}:{target_port}", flush=True)
    async with server:
        await server.serve_forever()


def _split(address: str) -> tuple[str, int]:
    host, _, port = address.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected host:port, got {address!r}")
    return host, int(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True, type=_split)
    parser.add_argument("--target", required=True, type=_split)
    args = parser.parse_args(argv)
    asyncio.run(serve(*args.listen, *args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
