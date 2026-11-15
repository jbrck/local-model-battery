"""The bridge must carry bytes both ways and survive a dead target."""
import asyncio

import unittest

from scripts.tools import endpoint_bridge as bridge


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while True:
        chunk = await reader.read(1024)
        if not chunk:
            break
        writer.write(chunk.upper())
        await writer.drain()
    writer.close()


class BridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.echo = await asyncio.start_server(_echo, "127.0.0.1", 0)
        self.echo_port = self.echo.sockets[0].getsockname()[1]
        self.relay = await asyncio.start_server(
            lambda r, w: bridge._handle(r, w, "127.0.0.1", self.echo_port), "127.0.0.1", 0)
        self.relay_port = self.relay.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.echo.close()
        self.relay.close()

    async def test_bytes_travel_both_ways(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.relay_port)
        writer.write(b"hello bridge")
        await writer.drain()
        self.assertEqual(await reader.readexactly(12), b"HELLO BRIDGE")
        writer.close()

    async def test_a_dead_target_returns_eof_instead_of_hanging(self):
        closed = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        closed_port = closed.sockets[0].getsockname()[1]
        closed.close()
        await closed.wait_closed()
        dead_relay = await asyncio.start_server(
            lambda r, w: bridge._handle(r, w, "127.0.0.1", closed_port), "127.0.0.1", 0)
        dead_port = dead_relay.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", dead_port)
            self.assertEqual(await asyncio.wait_for(reader.read(16), timeout=3), b"")
            writer.close()
        finally:
            dead_relay.close()


class SplitTest(unittest.TestCase):
    def test_host_port_splits(self):
        self.assertEqual(bridge._split("172.17.0.1:8124"), ("172.17.0.1", 8124))

    def test_a_missing_port_is_rejected(self):
        with self.assertRaises(Exception):
            bridge._split("172.17.0.1")
