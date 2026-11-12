"""Host facts: GPUs, memory, CPU and the external tools a leg may need.

Every probe reports a stated absence instead of raising. A run that dies because
nvidia-smi is missing tells you nothing about the model.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass

PROBE_TIMEOUT_S = 20
MEBIBYTE = 1024 * 1024
GPU_QUERY = "index,name,memory.total,memory.used,power.limit"
COMPUTE_APP_QUERY = "gpu_uuid,pid,process_name,used_memory"
VERSION_LINE_CHARS = 80
MEMINFO_PATH = "/proc/meminfo"
CPUINFO_PATH = "/proc/cpuinfo"
NUMA_GLOB = "/sys/devices/system/node/node[0-9]*"


@dataclass(frozen=True)
class ToolSpec:
    """An external binary a leg may call, and how to ask it for a version."""

    name: str
    version_args: tuple[str, ...] = ("--version",)
    env_var: str | None = None


# Tools whose presence changes what the battery can run. A host may override the binary
# with the named environment variable.
DEFAULT_TOOL_SPECS = (
    ToolSpec("docker", ("--version",), "DOCKER_BIN"),
    ToolSpec("tool-eval-bench", ("--version",), "TOOL_EVAL_BENCH_BIN"),
    ToolSpec("benchlocal-cli", ("--version",), "BENCHLOCAL_BIN"),
)


@dataclass(frozen=True)
class Gpu:
    """One accelerator, as nvidia-smi reports it."""

    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    power_limit_w: float | None = None

    @property
    def vram_free_mb(self) -> int:
        return max(0, self.vram_total_mb - self.vram_used_mb)

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    def to_dict(self) -> dict:
        return {
            "index": self.index, "name": self.name,
            "vram_total_mb": self.vram_total_mb, "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb, "power_limit_w": self.power_limit_w,
        }


@dataclass(frozen=True)
class ToolFact:
    """Whether a tool exists, where it is, and what version it reports."""

    name: str
    path: str | None = None
    version: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.path)

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available,
                "path": self.path, "version": self.version}


@dataclass(frozen=True)
class HostFacts:
    """Everything the planner needs to know about the machine it is on."""

    gpus: tuple[Gpu, ...] = ()
    gpu_error: str | None = None
    compute_apps: tuple[dict, ...] = ()
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    swap_used_gb: float | None = None
    cpu_model: str | None = None
    cpu_threads: int | None = None
    numa_nodes: int | None = None
    tools: tuple[ToolFact, ...] = ()

    @property
    def vram_total_gb(self) -> float:
        return sum(gpu.vram_total_gb for gpu in self.gpus)

    @property
    def vram_used_gb(self) -> float:
        return sum(gpu.vram_used_mb for gpu in self.gpus) / 1024

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def tool(self, name: str) -> ToolFact | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def to_dict(self) -> dict:
        return {
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "gpu_error": self.gpu_error,
            "vram_total_gb": round(self.vram_total_gb, 1),
            "vram_used_gb": round(self.vram_used_gb, 1),
            "compute_apps": list(self.compute_apps),
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "swap_used_gb": self.swap_used_gb,
            "cpu_model": self.cpu_model,
            "cpu_threads": self.cpu_threads,
            "numa_nodes": self.numa_nodes,
            "tools": [tool.to_dict() for tool in self.tools],
        }


def _run(command: list[str], timeout_s: int = PROBE_TIMEOUT_S,
         ) -> subprocess.CompletedProcess[str] | None:
    """Run a probe command, or return None when the binary is missing or hangs."""
    if not shutil.which(command[0]):
        return None
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False,
                              timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _optional_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _query_gpus() -> tuple[tuple[Gpu, ...], str | None]:
    """Accelerators, or the reason there are none."""
    result = _run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
    if result is None:
        return (), "nvidia-smi not found or did not answer"
    if result.returncode != 0:
        return (), (result.stderr or "nvidia-smi failed").strip()[:200] or "nvidia-smi failed"
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        gpus.append(Gpu(index=int(parts[0]), name=parts[1], vram_total_mb=int(parts[2]),
                        vram_used_mb=int(parts[3]), power_limit_w=_optional_float(parts[4])))
    return tuple(gpus), None if gpus else "nvidia-smi reported no devices"


def _query_compute_apps() -> tuple[dict, ...]:
    """Processes holding VRAM. Anything here competes with the model under test."""
    result = _run(["nvidia-smi", f"--query-compute-apps={COMPUTE_APP_QUERY}",
                   "--format=csv,noheader,nounits"])
    if result is None or result.returncode != 0:
        return ()
    apps = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append({"gpu_uuid": parts[0], "pid": parts[1],
                     "process_name": parts[2], "used_memory_mb": _optional_float(parts[3])})
    return tuple(apps)


def _read_meminfo() -> tuple[float | None, float | None, float | None]:
    """(total GB, available GB, swap used GB) from /proc/meminfo."""
    if not os.path.isfile(MEMINFO_PATH):
        return None, None, None
    values: dict[str, float] = {}
    with open(MEMINFO_PATH, encoding="utf-8") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            fields = rest.split()
            if fields and fields[0].isdigit():
                values[name.strip()] = int(fields[0]) / MEBIBYTE
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_used = None
    if "SwapTotal" in values and "SwapFree" in values:
        swap_used = round(values["SwapTotal"] - values["SwapFree"], 1)
    return (round(total, 1) if total else None,
            round(available, 1) if available else None,
            swap_used)


def _read_cpu() -> tuple[str | None, int | None]:
    """CPU model string and thread count."""
    model = None
    if os.path.isfile(CPUINFO_PATH):
        with open(CPUINFO_PATH, encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    model = line.partition(":")[2].strip()
                    break
    return model, os.cpu_count()


def _count_numa_nodes() -> int | None:
    nodes = glob.glob(NUMA_GLOB)
    return len(nodes) or None


def _probe_tool(spec: ToolSpec) -> ToolFact:
    """Locate a tool (environment override first) and record the version it reports."""
    override = os.environ.get(spec.env_var, "") if spec.env_var else ""
    path = override or shutil.which(spec.name) or ""
    if not path:
        return ToolFact(spec.name)
    result = _run([path, *spec.version_args])
    version = None
    if result is not None and result.returncode == 0:
        lines = (result.stdout or result.stderr).strip().splitlines()
        version = lines[0][:VERSION_LINE_CHARS] if lines else None
    return ToolFact(spec.name, path, version)


def probe_host(tool_specs: tuple[ToolSpec, ...] = DEFAULT_TOOL_SPECS) -> HostFacts:
    """Probe the machine. Runs outside the sandbox, so permissions and paths are real."""
    gpus, gpu_error = _query_gpus()
    ram_total, ram_available, swap_used = _read_meminfo()
    cpu_model, cpu_threads = _read_cpu()
    return HostFacts(
        gpus=gpus,
        gpu_error=gpu_error,
        compute_apps=_query_compute_apps() if gpus else (),
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        swap_used_gb=swap_used,
        cpu_model=cpu_model,
        cpu_threads=cpu_threads,
        numa_nodes=_count_numa_nodes(),
        tools=tuple(_probe_tool(spec) for spec in tool_specs),
    )
