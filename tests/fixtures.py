"""Hosts, endpoints and profiles the planner tests share.

The point of the battery is that the hosts differ, so the fixtures are two real shapes: one
24 GB card with everything installed, and a pair of 20 GB cards with nothing installed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.engine_facts import EngineFacts  # noqa: E402
from scripts.lib.env_probe import Gpu, HostFacts, ToolFact  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.model_probe import ModelShape  # noqa: E402
from scripts.lib.pilot import PilotResult  # noqa: E402
from scripts.lib.plan_run import RunPlan, plan_run  # noqa: E402
from scripts.lib.prereqs import Judge  # noqa: E402
from scripts.lib.server_probe import ServerFacts  # noqa: E402
from scripts.lib.variables import load_variables  # noqa: E402

STANDARD = load_toml("standard.toml")
LEGS = load_legs(STANDARD)
BY_ID = {leg.id: leg for leg in LEGS}
VARIABLES = load_variables()
TOOLS = (ToolFact("docker", "/usr/bin/docker", "27.0"),
         ToolFact("benchlocal-cli", "/usr/local/bin/benchlocal-cli", "0.4"),
         ToolFact("tool-eval-bench", "/usr/local/bin/tool-eval-bench", "1.2"))

# 3090: one 24 GB card, 64 GB of RAM, every external tool present.
RICH = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 0, 350.0),), ram_total_gb=64.0,
                 ram_available_gb=60.0, cpu_model="i7-8700K", cpu_threads=12, tools=TOOLS)
# 3080 x2: 40 GB of VRAM in total but 20 GB per card, and nothing else installed.
PAIR = HostFacts(gpus=(Gpu(0, "RTX 3080", 20480, 0, 320.0), Gpu(1, "RTX 3080", 20480, 0, 320.0)),
                 ram_total_gb=96.0, ram_available_gb=88.0, tools=())
BAREMETAL = HostFacts(ram_total_gb=32.0, ram_available_gb=28.0, gpu_error="nvidia-smi not found")

SERVER = ServerFacts(reachable=True, served_model="arete-27b", ctx=131072, slots=4,
                     chat_template_mentions_thinking=True, thinking_switch=True,
                     reasoning_channel=True, thinking_on_tokens=1400, thinking_off_tokens=90,
                     tools_with_json_schema_ok=True, props={"build_info": "b4567-abc",
                                                            "model_path": "arete-27b.gguf"})
# The engine behind SERVER, as its props probe would identify it.
ENGINE = EngineFacts(engine="llama.cpp", version="b4567-abc", window=131072, slots=4,
                     model_path="arete-27b.gguf", supports_kv_flags=True, source="props")
# The same endpoint served by vllm: version and max_model_len, no cache or attention flags.
VLLM = EngineFacts(engine="vllm", version="0.6.6", window=131072, source="version + models")
DEAF = ServerFacts(reachable=True, served_model="arete-27b", ctx=131072, slots=1,
                   chat_template_mentions_thinking=False, thinking_switch=False,
                   reasoning_channel=False, thinking_on_tokens=300, thinking_off_tokens=300,
                   tools_with_json_schema_ok=False, tools_error="grammar and json_schema: 400",
                   props={"build_info": "b11670"})
PILOT = PilotResult(tps=27.0, latency_s=1.2, generated_tokens=810, requests=3, failed=0,
                    stalled=False)
DEAD_PILOT = PilotResult(tps=0.0, latency_s=0.0, generated_tokens=0, requests=3, failed=3,
                         stalled=True, errors=("HTTP 500",))

# 27B dense Q4_K_M: 64 layers, 8 KV heads of 128, q8_0 KV.
FITS = {"id": "arete-27b", "weights_gb": 16.4, "thinking": "on", "weight_quant": "Q4_K_M",
        "model": {"n_layers": 64, "n_kv_heads": 8, "head_dim": 128}}
SMALL = {"id": "tiny-8b", "weights_gb": 5.2, "thinking": "on",
         "model": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}}
SHAPELESS = {"id": "mystery-30b", "weights_gb": 18.0}
JUDGE = Judge(model="deckard-40b", endpoint="http://127.0.0.1:11435/v1", under_test="arete-27b")
# Two legs with no external prerequisites, for the tests that need a quiet host.
FREE = ("math500", "mmlu")


def plan(profile: dict = FITS, standard: dict | None = None, host: HostFacts = RICH,
         server: ServerFacts = SERVER, pilot: PilotResult = PILOT,
         judge: Judge | None = JUDGE, selected: tuple[str, ...] = (),
         shape: ModelShape | None = None, serving_pid: int | None = None,
         engine: EngineFacts | None = ENGINE) -> RunPlan:
    return plan_run(standard or STANDARD, LEGS, VARIABLES, profile, host, server, pilot,
                    judge, selected, model_shape=shape, serving_pid=serving_pid, engine=engine)


def runnable(result: RunPlan, leg_id: str):
    """The planned leg, or a failed test when the planner dropped it."""
    leg = result.leg(leg_id)
    assert leg is not None, f"{leg_id} is not in this plan"
    return leg


def decision(result: RunPlan, variable: str):
    """The one decision row for a variable."""
    rows = [item for item in result.decisions if item.variable == variable]
    assert len(rows) == 1, f"expected one row for {variable}, found {len(rows)}"
    return rows[0]
