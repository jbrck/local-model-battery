"""The consult brief: sections present, tables read by id, unknowns stated.

The brief is rendered from facts, so the assertions are on stable things -- the eight
section headings, a row's own serve text read back from engines.toml, a judge candidate
that must be excluded -- never on wording the brief happens to use. HTTP is stubbed the
same way the rest of the suite stubs it: a local JsonStub, no external calls.
"""
from __future__ import annotations

import argparse
import io
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib import consult  # noqa: E402
from scripts.lib.consult import (  # noqa: E402
    BriefInputs,
    MissingKnowledge,
    SMOKE_CONCRETE,
    _as_int,
    _call_verdict,
    base_url,
    detect_model_types,
    engine_candidates,
    engine_row,
    fetch_model_ids,
    judge_candidates,
    lineup_command,
    load_knowledge,
    probe_engine_or_none,
    render_brief,
    render_call,
    render_decisions,
    render_engine,
    render_host,
    render_judge,
    render_model,
    rows_by_id,
)
from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import Gpu, HostFacts, ToolFact  # noqa: E402
from scripts.lib.fit import Fit  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.model_probe import (  # noqa: E402
    SOURCE_GGUF,
    SOURCE_HF,
    SOURCE_UNKNOWN,
    ModelShape,
)
from scripts.lib.variables import Variable  # noqa: E402
from tests.stub_json import JsonStub  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The measured-engine stand-in: the fields the brief reads off an EngineFacts object.
class StubEngineFacts:
    """Duck-typed EngineFacts: the same field names, fixed values."""

    engine = "llama.cpp"
    version = "b10909"
    window = 131072
    slots = 4
    kv_type = "q8_0"
    flash_attn = True
    drafter = None
    source = "props"
    errors: tuple[str, ...] = ()
    supports_kv_flags = True

    @property
    def known(self) -> bool:
        return True


UNKNOWN_ENGINE_FACTS = type("SilentEngine", (), {
    "engine": "unknown", "version": "", "window": None, "slots": None,
    "kv_type": None, "flash_attn": None, "drafter": None, "source": "",
    "errors": ("props: connection refused",), "supports_kv_flags": False,
    "known": False,
})()

OPENAI_COMPATIBLE_FACTS = type("UnidentifiedEndpointFacts", (), {
    "engine": "openai-compatible", "version": "", "window": None, "slots": None,
    "kv_type": None, "flash_attn": None, "drafter": None, "source": "models",
    "errors": ("props: 404", "version: 404"), "supports_kv_flags": False,
    "known": True,
})()

ENGINES = rows_by_id(load_knowledge(consult.ENGINES_FILE, consult.ENGINE_TABLE_KEY))


def brief_inputs(**overrides) -> BriefInputs:
    """A first-pass brief: no endpoint, no model file, a host with no facts."""
    host = overrides.pop("host", HostFacts())
    shape = overrides.pop(
        "shape",
        ModelShape(source=SOURCE_UNKNOWN,
                   problems=("no model file and no profile, so the shape is unknown",)))
    return BriefInputs(host=host, shape=shape, **overrides)


def section(text: str, number: int) -> str:
    """One numbered section's body, split out of the rendered brief."""
    start = text.index(f"## {number}. ")
    next_heading = text.find(f"\n## {number + 1}. ", start)
    return text[start:next_heading if next_heading != -1 else len(text)]


class BriefStructure(unittest.TestCase):
    """The first-pass case: every section present, unknowns stated, nothing decided."""

    def test_first_pass_brief_has_all_eight_sections(self) -> None:
        text = render_brief(brief_inputs())
        for number, heading in enumerate(consult.SECTION_HEADINGS, start=1):
            self.assertIn(f"## {number}. {heading}", text)
        self.assertLess(text.index("## 1. "), text.index("## 2. "))

    def test_first_pass_states_unknowns(self) -> None:
        text = render_brief(brief_inputs())
        self.assertIn("unknown", section(text, 1))   # accelerators: nvidia-smi said nothing
        self.assertIn("unknown", section(text, 2))   # no model file
        self.assertIn("unknown", section(text, 6))   # no endpoint to list judges from

    def test_first_pass_withholds_the_lineup_with_a_reason(self) -> None:
        text = render_brief(brief_inputs())
        lineup = section(text, 8)
        self.assertNotIn("run_battery.py", lineup)
        self.assertIn("Withheld", lineup)
        self.assertIn("section 3", lineup)

    def test_brief_never_decides(self) -> None:
        """The lineup's legs and judge are placeholders, not choices made for the user."""
        endpoint_inputs = brief_inputs(
            endpoint="http://127.0.0.1:9/v1", model_ids=("served-model",),
            model_name="served-model", engine_facts=StubEngineFacts())
        lineup = section(render_brief(endpoint_inputs), 8)
        self.assertIn("--legs <the leg set of the use case you choose", lineup)
        self.assertIn("--judge-model <the judge you choose", lineup)


class TablesAreReadById(unittest.TestCase):
    """A corrected row changes the brief with no code change; a missing one fails loudly."""

    def test_engine_row_by_id(self) -> None:
        row = engine_row("llama.cpp", ENGINES)
        self.assertTrue(row.get("serve"))

    def test_unknown_engine_row_fails_loudly(self) -> None:
        with self.assertRaises(MissingKnowledge):
            engine_row("not-an-engine", ENGINES)

    def test_missing_table_fails_loudly(self) -> None:
        with self.assertRaises(MissingKnowledge):
            load_knowledge("no-such-table.toml", consult.ENGINE_TABLE_KEY)

    def test_no_endpoint_with_engine_renders_that_rows_serve_and_smoke(self) -> None:
        row = engine_row("mlx", ENGINES)
        text = render_brief(brief_inputs(engine_id="mlx"))
        engine_section = section(text, 3)
        self.assertIn(row["serve"], engine_section)
        self.assertIn(row["smoke"], engine_section)
        self.assertIn("Nothing is up yet", engine_section)

    def test_no_endpoint_without_engine_lists_candidates(self) -> None:
        text = section(render_brief(brief_inputs()), 3)
        for candidate in engine_candidates(SOURCE_UNKNOWN, tuple(ENGINES)):
            self.assertIn(candidate, text)
        self.assertIn("--engine", text)

    def test_use_cases_rendered_from_the_table(self) -> None:
        use_cases = rows_by_id(load_knowledge(consult.USE_CASES_FILE,
                                              consult.USE_CASE_TABLE_KEY))
        self.assertEqual(len(use_cases), 4)
        text = section(render_brief(brief_inputs()), 5)
        for use_case_id in use_cases:
            self.assertIn(f"### {use_case_id}", text)

    def test_named_use_case_is_marked_not_chosen(self) -> None:
        text = section(render_brief(brief_inputs(use_case_id="ranking")), 5)
        self.assertIn("### ranking (you named this one)", text)


class EngineCandidates(unittest.TestCase):
    """Who could serve this file, decided by the tables, never by the operating system."""

    def test_gguf_goes_to_the_gguf_engines(self) -> None:
        candidates = engine_candidates(SOURCE_GGUF, tuple(ENGINES))
        self.assertEqual(candidates, ("llama.cpp", "ollama"))

    def test_hf_shape_excludes_the_gguf_engines(self) -> None:
        candidates = engine_candidates(SOURCE_HF, tuple(ENGINES))
        self.assertIn("vllm", candidates)
        self.assertNotIn("llama.cpp", candidates)
        self.assertNotIn("proxy", candidates)

    def test_unknown_shape_rules_nothing_out(self) -> None:
        candidates = engine_candidates(SOURCE_UNKNOWN, tuple(ENGINES))
        self.assertIn("llama.cpp", candidates)
        self.assertIn("mlx", candidates)
        self.assertNotIn("proxy", candidates)

    def test_mlx_is_an_ordinary_no_endpoint_case(self) -> None:
        row = engine_row("mlx", ENGINES)
        text = render_brief(brief_inputs(engine_id="mlx"))
        self.assertIn(row["serve"], section(text, 3))
        self.assertIn("setup checklist", section(text, 7))


class ModelTypes(unittest.TestCase):
    """What the shape can and cannot say about the model's type."""

    def test_hybrid_notes_match_hybrid_recurrent(self) -> None:
        shape = ModelShape(source=SOURCE_GGUF, n_layers=16,
                           notes=("16 of 64 layers are attention layers; the rest are "
                                  "recurrent and hold no growing cache",))
        matched, _ = detect_model_types(shape)
        self.assertIn("hybrid_recurrent", matched)
        self.assertNotIn("dense", matched)

    def test_plain_gguf_shape_is_dense(self) -> None:
        matched, _ = detect_model_types(ModelShape(source=SOURCE_GGUF, n_layers=32))
        self.assertIn("dense", matched)

    def test_quant_tag_in_the_path_matches_quantized(self) -> None:
        matched, _ = detect_model_types(
            ModelShape(source=SOURCE_GGUF, n_layers=32, path="models/Q4_K_M/model.gguf"))
        self.assertIn("quantized", matched)

    def test_plain_name_does_not_read_as_a_quant(self) -> None:
        matched, _ = detect_model_types(
            ModelShape(source=SOURCE_GGUF, n_layers=32, path="models/arete-27b/model.gguf"))
        self.assertNotIn("quantized", matched)

    def test_large_declared_window_is_long_context(self) -> None:
        matched, _ = detect_model_types(
            ModelShape(source=SOURCE_GGUF, n_layers=32, native_ctx=131072))
        self.assertIn("long_context", matched)

    def test_moe_and_reasoning_stay_unresolved(self) -> None:
        _matched, unresolved = detect_model_types(
            ModelShape(source=SOURCE_GGUF, n_layers=32))
        self.assertIn("sparse_moe", unresolved)
        self.assertIn("reasoning", unresolved)
        text = render_brief(brief_inputs(shape=ModelShape(source=SOURCE_GGUF, n_layers=32,
                                                          weights_bytes=1024,
                                                          kv_elements_per_token=1024)))
        self.assertIn("`sparse_moe`: unknown", section(text, 2))
        self.assertIn("`reasoning`: unknown", section(text, 2))


class JudgeSection(unittest.TestCase):
    """Candidates come from the endpoint, minus everything under test."""

    def test_candidates_exclude_the_model_under_test(self) -> None:
        candidates = judge_candidates(("judge-a", "under-test", "judge-b"), {"under-test"})
        self.assertEqual([model for model, _pin in candidates], ["judge-a", "judge-b"])

    def test_endpoint_list_through_the_stub(self) -> None:
        routes = {"/v1/models": (200, {"data": [{"id": "under-test"},
                                                {"id": "judge-a"},
                                                {"id": "nous-glm-5-3-flash"}]})}
        with JsonStub(routes) as stub:
            ids, error = fetch_model_ids(ModelConfig(name="", endpoint=stub.endpoint))
        self.assertIsNone(error)
        candidates = judge_candidates(ids, {"under-test"})
        self.assertEqual([model for model, _pin in candidates],
                         ["judge-a", "nous-glm-5-3-flash"])

    def test_pinned_default_is_named_out_loud(self) -> None:
        pin = load_toml(consult.STANDARD_FILE).get("judge") or {}
        text = render_brief(brief_inputs(model_ids=("served-model",),
                                         model_name="served-model"))
        self.assertIn(pin["model"], section(text, 6))

    def test_under_test_never_appears_as_a_candidate(self) -> None:
        text = render_brief(brief_inputs(model_ids=("served-model", "judge-a"),
                                         model_name="served-model"))
        judge_section = section(text, 6)
        self.assertIn("Excluded", judge_section)
        self.assertIn("served-model", judge_section)
        self.assertIn("- judge-a", judge_section)
        self.assertNotIn("- served-model", judge_section)


class EngineFactsRendering(unittest.TestCase):
    """A reachable endpoint renders measured facts instead of serve instructions."""

    def test_measured_facts_reach_the_brief(self) -> None:
        text = render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                         engine_facts=StubEngineFacts()))
        engine_section = section(text, 3)
        self.assertIn("llama.cpp", engine_section)
        self.assertIn("b10909", engine_section)
        self.assertIn("131072 tokens", engine_section)
        self.assertNotIn("Nothing is up yet", engine_section)

    def test_reachable_but_silent_engine_is_unknown(self) -> None:
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 engine_facts=UNKNOWN_ENGINE_FACTS)), 3)
        self.assertIn("unknown", text)
        self.assertIn("props: connection refused", text)

    def test_an_identified_engine_does_not_list_the_probes_it_does_not_serve(self) -> None:
        facts = type("ProbedFacts", (), {
            "engine": "llama.cpp", "version": "b10909", "window": 131072,
            "slots": 1, "kv_type": "q8_0", "flash_attn": True, "drafter": None,
            "source": "props", "known": True, "supports_kv_flags": True,
            "errors": ("models: 404", "version: 404"),
        })()
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 engine_facts=facts)), 3)
        self.assertIn("2 other probes were not served", text)
        self.assertNotIn("probe error", text)

    def test_engine_with_no_table_row_says_so(self) -> None:
        facts = type("UnknownEngineFacts", (), {
            "engine": "engine-from-the-future", "version": "", "window": 8192,
            "source": "models", "errors": (), "known": True,
        })()
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 engine_facts=facts)), 4)
        self.assertIn("unknown", text)

    def test_the_unidentified_endpoint_has_a_row_and_a_smoke_recipe(self) -> None:
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 engine_facts=OPENAI_COMPATIBLE_FACTS)), 4)
        self.assertNotIn("no engines.toml row", text)
        self.assertIn("not what the model does", text)
        request, passes = SMOKE_CONCRETE["openai-compatible"]
        self.assertIn("/models", request.format(base="http://127.0.0.1:9/v1"))
        self.assertIn("completion", passes)

    def test_every_engine_row_can_be_compared_or_says_it_cannot(self) -> None:
        for row in ENGINES.values():
            self.assertTrue(row.get("compares_to"), row.get("id"))


class TheCall(unittest.TestCase):
    """Section 4 is a verdict with a reason, never silence."""

    def test_no_model_shape_prints_unknown(self) -> None:
        text = section(render_brief(brief_inputs()), 4)
        self.assertIn("unknown", text)

    def test_fitting_model_gets_a_verdict(self) -> None:
        gpu = Gpu(index=0, name="Test Card", vram_total_mb=24576, vram_used_mb=0)
        shape = ModelShape(source=SOURCE_GGUF, n_layers=32, n_kv_heads=8, head_dim=128,
                           kv_elements_per_token=2 * 32 * 8 * 128,
                           weights_bytes=16 * 1024 ** 3, native_ctx=131072,
                           path="models/Q4_K_M/model.gguf")
        text = section(render_brief(brief_inputs(host=HostFacts(gpus=(gpu,)),
                                                 shape=shape)), 4)
        self.assertIn("the call:", text)
        self.assertNotIn("unknown -- no model shape", text)


class SmokeAndLineup(unittest.TestCase):
    """Section 7 is concrete; section 8 is the exact command when something is up."""

    def test_smoke_is_concrete_with_an_endpoint(self) -> None:
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 engine_facts=StubEngineFacts())), 7)
        self.assertIn("GET http://127.0.0.1:9/props", text)
        self.assertIn("a pass looks like", text)
        self.assertIn("Stop and fix", text)

    def test_setup_checklist_without_an_endpoint(self) -> None:
        text = section(render_brief(brief_inputs(engine_id="llama.cpp")), 7)
        self.assertIn("setup checklist", text)
        self.assertIn(engine_row("llama.cpp", ENGINES)["serve"], text)

    def test_lineup_names_the_real_entry_point(self) -> None:
        text = section(render_brief(brief_inputs(endpoint="http://127.0.0.1:9/v1",
                                                 model_ids=("served-model",),
                                                 model_name="served-model",
                                                 engine_facts=StubEngineFacts())), 8)
        self.assertIn("python3 scripts/run_battery.py --model served-model "
                      "--endpoint http://127.0.0.1:9/v1", text)
        self.assertIn("--detach", text)
        self.assertIn("--status", text)

    def test_lineup_command_is_exact(self) -> None:
        command = lineup_command("m", "http://h:1/v1", "m.gguf", "", "all", "j", "", "runs/m")
        self.assertEqual(
            command,
            "python3 scripts/run_battery.py --model m --endpoint http://h:1/v1 "
            "--model-file m.gguf --legs all --judge-model j --out runs/m\n"
            "python3 scripts/run_battery.py --model m --endpoint http://h:1/v1 "
            "--model-file m.gguf --legs all --judge-model j --out runs/m --detach\n"
            "python3 scripts/run_battery.py --out runs/m --status")


class Helpers(unittest.TestCase):
    """Small pure pieces the brief is built from."""

    def test_base_url_strips_the_v1_suffix(self) -> None:
        self.assertEqual(base_url("http://h:8080/v1/"), "http://h:8080")
        self.assertEqual(base_url("http://h:8080"), "http://h:8080")


class CommandLine(unittest.TestCase):
    """The real command renders a brief on this bare host, with no endpoint."""

    def test_scripts_consult_prints_the_brief(self) -> None:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "consult.py")],
            capture_output=True, text=True, timeout=120, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        for number in range(1, 9):
            self.assertIn(f"## {number}. ", result.stdout)
        self.assertIn("Nothing is up yet", result.stdout)

    def test_unknown_engine_id_fails_loudly(self) -> None:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "consult.py"),
             "--engine", "not-an-engine"],
            capture_output=True, text=True, timeout=120, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no row for 'not-an-engine'", result.stderr)


class FetchModelIds(unittest.TestCase):
    """The endpoint's model list, or the reason there is none: every failure named."""

    CFG = ModelConfig(name="", endpoint="http://h:9/v1")

    def ids_with_payload(self, payload: object, error: str | None = None):
        with mock.patch("scripts.lib.consult.get_json",
                        return_value=(200, payload, error)):
            return fetch_model_ids(self.CFG)

    def test_a_transport_error_comes_back_verbatim(self):
        ids, error = self.ids_with_payload(None, error="connection refused")
        self.assertEqual(ids, ())
        self.assertEqual(error, "connection refused")

    def test_a_payload_that_is_not_an_object_is_not_a_model_list(self):
        ids, error = self.ids_with_payload("not json")
        self.assertEqual((ids, error), ((), "the model list is not a JSON array of entries"))

    def test_a_data_field_that_is_not_an_array_is_not_a_model_list(self):
        ids, error = self.ids_with_payload({"data": {"id": "x"}})
        self.assertEqual((ids, error), ((), "the model list is not a JSON array of entries"))

    def test_entries_without_an_id_are_skipped_not_fatal(self):
        ids, error = self.ids_with_payload(
            {"data": [{"id": "a"}, {"other": 1}, "junk", {"id": ""}]})
        self.assertEqual(ids, ("a",))
        self.assertIsNone(error)

    def test_an_empty_list_says_so(self):
        ids, error = self.ids_with_payload({"data": []})
        self.assertEqual((ids, error), ((), "the model list is empty"))

    def test_a_good_list_comes_back_in_order(self):
        ids, error = self.ids_with_payload({"data": [{"id": "one"}, {"id": "two"}]})
        self.assertEqual(ids, ("one", "two"))
        self.assertIsNone(error)


class ProbeEngineOrNone(unittest.TestCase):
    """The measured engine when the engine-facts module is present, None when not."""

    CFG = ModelConfig(name="", endpoint="http://h:9/v1")

    def test_without_the_module_the_brief_gets_no_engine_facts(self):
        with mock.patch("scripts.lib.consult.ENGINE_FACTS_AVAILABLE", False), \
                mock.patch("scripts.lib.consult.probe_engine", None):
            self.assertIsNone(probe_engine_or_none(self.CFG))

    def test_with_the_module_the_probe_runs_and_its_result_is_kept(self):
        facts = SimpleNamespace(engine="llama.cpp")
        with mock.patch("scripts.lib.consult.ENGINE_FACTS_AVAILABLE", True), \
                mock.patch("scripts.lib.consult.probe_engine",
                           return_value=facts) as probe:
            result = probe_engine_or_none(self.CFG)
        self.assertIs(result, facts)
        probe.assert_called_once_with(self.CFG)


class AsInt(unittest.TestCase):
    """Config-table numbers: a bool is not an int, whatever Python thinks."""

    def test_an_int_comes_back_as_itself(self):
        self.assertEqual(_as_int(131072, 1), 131072)

    def test_a_bool_is_rejected(self):
        self.assertEqual(_as_int(True, 5), 5)

    def test_anything_else_takes_the_default(self):
        self.assertEqual(_as_int("big", 7), 7)
        self.assertEqual(_as_int(None, 7), 7)


class HostSection(unittest.TestCase):
    """Section 1 renders every fact it was given, each naming its source."""

    RICH = HostFacts(
        gpus=(Gpu(0, "RTX 3090", 24576, 0, 350.0),),
        ram_total_gb=64.0, ram_available_gb=60.0, swap_used_gb=2.0,
        cpu_model="i7-8700K", cpu_threads=12, numa_nodes=2,
        tools=(ToolFact("docker", "/usr/bin/docker", "27.0"),
               ToolFact("benchlocal-cli", None, None)))

    def test_a_full_host_renders_every_line(self):
        text = render_host(self.RICH)
        self.assertIn("- accelerator 0: RTX 3090, 24.0 GB total, 24.0 GB free", text)
        self.assertIn("- memory: 64.0 GB total, 60.0 GB available, 2.0 GB swap in use", text)
        self.assertIn("- cpu: i7-8700K, 12 threads", text)
        self.assertIn("- numa nodes: 2", text)
        self.assertIn("- tool docker: present at /usr/bin/docker, reports 27.0", text)

    def test_an_absent_tool_names_the_cost(self):
        text = render_host(self.RICH)
        self.assertIn("- tool benchlocal-cli: absent", text)
        self.assertIn("skipped", text)

    def test_a_tool_without_a_version_omits_the_report(self):
        host = HostFacts(tools=(ToolFact("docker", "/usr/bin/docker", None),))
        self.assertIn("- tool docker: present at /usr/bin/docker (source:", render_host(host))

    def test_a_host_with_no_ram_or_cpu_says_unknown(self):
        text = render_host(HostFacts())
        self.assertIn("memory: unknown", text)
        self.assertIn("cpu: unknown", text)

    def test_a_cpu_with_threads_but_no_model_says_model_not_reported(self):
        text = render_host(HostFacts(cpu_threads=8))
        self.assertIn("- cpu: model not reported, 8 threads", text)

    def test_no_numa_line_when_the_kernel_exposes_one_node(self):
        self.assertNotIn("numa", render_host(HostFacts(cpu_threads=1)))


class ModelSection(unittest.TestCase):
    """Section 2 reads its types from the table and fails loudly on a missing row."""

    KNOWN = ModelShape(source=SOURCE_GGUF, n_layers=32, n_kv_heads=8, head_dim=128,
                       kv_elements_per_token=65536, weights_bytes=16 * 1024 ** 3,
                       path="models/arete-27b/model.gguf")

    def test_a_missing_model_type_row_fails_loudly(self):
        with self.assertRaises(MissingKnowledge) as caught:
            render_model(self.KNOWN, {})
        self.assertIn("dense", str(caught.exception))

    def test_a_known_shape_names_the_unresolvable_types_too(self):
        text = render_model(self.KNOWN, {"dense": {"effect": "e", "advice": "a"}})
        self.assertIn("`dense`: e", text)
        self.assertIn("records: a", text)
        self.assertIn("`sparse_moe`: unknown", text)
        self.assertIn("`reasoning`: unknown", text)

    def test_an_unknown_shape_lists_its_problems(self):
        shape = ModelShape(problems=("no header", "truncated file"))
        text = render_model(shape, {})
        self.assertIn("- problem: no header", text)
        self.assertIn("- problem: truncated file", text)


class EngineSection(unittest.TestCase):
    """Section 3 has three sources of truth: measured, endpoint-only, and nothing up."""

    def test_an_endpoint_without_the_facts_module_says_what_would_resolve_it(self):
        text = render_engine(brief_inputs(endpoint="http://h:9/v1"), ENGINES)
        self.assertIn("unknown", text)
        self.assertIn("engine-facts module", text)

    def test_a_drafter_the_engine_reports_is_named(self):
        facts = type("DraftedFacts", (), {
            "engine": "strata", "version": "v9", "window": 32768, "slots": None,
            "kv_type": None, "flash_attn": None, "drafter": "mtp", "source": "health",
            "errors": (), "known": True, "supports_kv_flags": False,
        })()
        text = render_engine(brief_inputs(endpoint="http://h:9/v1", engine_facts=facts),
                             ENGINES)
        self.assertIn("- drafter: mtp", text)

    def test_an_engine_facts_object_without_the_attribute_still_renders(self):
        """Duck typing: the brief reads what is there and leaves the rest out."""
        text = render_engine(brief_inputs(endpoint="http://h:9/v1",
                                          engine_facts=StubEngineFacts()), ENGINES)
        self.assertIn("llama.cpp", text)


class CallVerdict(unittest.TestCase):
    """One sentence on whether the combination is worth testing, for every outcome."""

    FIT = Fit(tier="T1", vram_total_gb=24.0, vram_free_gb=20.0, weights_gb=16.0,
              vram_needed_gb=18.0, headroom_gb=2.0, max_ctx=65536, preferred_ctx=131072,
              kv_bytes_per_token=131072, reason="16 GB of weights on a 20 GB card")

    def verdict(self, fit, window, preferred_ctx=131072):
        facts = SimpleNamespace(window=window) if window is not Ellipsis else None
        return _call_verdict(fit, facts, preferred_ctx)

    def test_no_shape_means_an_unknown_call(self):
        self.assertIn("unknown", self.verdict(None, None))

    def test_a_model_that_cannot_fit_is_not_worth_testing(self):
        no_fit = Fit(tier="T4", vram_total_gb=8.0, vram_free_gb=4.0, weights_gb=16.0,
                     vram_needed_gb=20.0, headroom_gb=-2.0, max_ctx=0,
                     preferred_ctx=131072, kv_bytes_per_token=131072, reason="it will not fit")
        text = self.verdict(no_fit, None)
        self.assertIn("not worth testing", text)
        self.assertIn("it will not fit", text)

    def test_a_smaller_served_window_is_worth_testing_at_a_reduction(self):
        text = self.verdict(self.FIT, 65536)
        self.assertIn("reduced window", text)
        self.assertIn("65536", text)

    def test_a_window_that_reaches_the_standard_is_worth_testing(self):
        text = self.verdict(self.FIT, 131072)
        self.assertIn("the window reaches the standard", text)

    def test_an_unknown_window_waits_for_the_smoke_check(self):
        text = self.verdict(self.FIT, None)
        self.assertIn("once the engine is up", text)
        self.assertIn("smoke check", text)


class CallSection(unittest.TestCase):
    """Section 4 renders the fit, the window, the engine row and the call."""

    def test_an_estimate_fit_says_the_placement_is_the_engines_call(self):
        fit = Fit(tier="T1", vram_total_gb=24.0, vram_free_gb=20.0, weights_gb=16.0,
                  vram_needed_gb=18.0, headroom_gb=2.0, max_ctx=131072,
                  preferred_ctx=131072, kv_bytes_per_token=131072, reason="fits",
                  estimate=True)
        text = render_call(brief_inputs(endpoint="http://h:9/v1",
                                        engine_facts=StubEngineFacts()), fit, ENGINES,
                           131072)
        self.assertIn("the fit is an estimate", text)
        self.assertIn("what llama.cpp does to the numbers", text)
        self.assertIn("window: the engine serves 131072 tokens", text)

    def test_a_kv_less_engine_says_the_knob_is_not_there(self):
        fit = Fit(tier="T1", vram_total_gb=24.0, vram_free_gb=20.0, weights_gb=5.0,
                  vram_needed_gb=6.0, headroom_gb=14.0, max_ctx=131072,
                  preferred_ctx=131072, kv_bytes_per_token=131072, reason="fits")
        text = render_call(brief_inputs(endpoint="http://h:9/v1",
                                        engine_facts=OPENAI_COMPATIBLE_FACTS), fit,
                           ENGINES, 131072)
        self.assertIn("KV cache is not a knob", text)


class DecisionsSection(unittest.TestCase):
    """Section 5 lists the use cases and the five variables, or fails loudly."""

    USE_CASES = {"ranking": {"question": "q", "legs": "l", "cannot_say": "c"}}

    def variable(self, variable_id):
        return Variable(id=variable_id, variable_class="M", applies_to=("all",),
                        stamp_key=variable_id, what_it_changes="changes it",
                        effect="measured", source="the table", decision_rule="decide")

    def variables(self, *ids):
        return {variable_id: self.variable(variable_id) for variable_id in ids}

    def test_every_decision_variable_needs_its_row(self):
        inputs = brief_inputs(use_case_id="ranking")
        variables = self.variables(*(variable_id for _label, variable_id
                                     in consult.DECISION_VARIABLES))
        text = render_decisions(inputs, self.USE_CASES, variables)
        self.assertIn("### ranking (you named this one)", text)
        self.assertIn("- cap (measurement): changes it -- decide", text)
        self.assertIn("recorded in the run stamp", text)

    def test_a_missing_variable_row_is_a_loud_failure(self):
        inputs = brief_inputs()
        with self.assertRaises(MissingKnowledge) as caught:
            render_decisions(inputs, self.USE_CASES, {})
        self.assertIn("variables.toml has no row for 'cap'", str(caught.exception))


class JudgeMarkers(unittest.TestCase):
    """The marker after a candidate: the operator's pick outranks the pinned default."""

    def inputs_with(self, model_ids, judge_pick="", model_name=""):
        return brief_inputs(model_ids=model_ids, model_name=model_name,
                            judge_pick=judge_pick)

    def test_the_operators_pick_is_marked_over_the_pin(self):
        text = render_judge(self.inputs_with(("deckard", "pick"), judge_pick="pick"),
                            {"model": "deckard"})
        self.assertIn("- deckard (the pinned default)", text)
        self.assertIn("- pick (you named this one)", text)
        self.assertIn("You named pick as the judge.", text)

    def test_a_candidate_without_a_marker_stays_plain(self):
        text = render_judge(self.inputs_with(("other",)), {"model": "none"})
        self.assertIn("- other\n", text)

    def test_a_pin_missing_from_the_endpoint_is_named(self):
        text = render_judge(self.inputs_with(("other",)), {"model": "deckard"})
        self.assertIn("The pinned default deckard is not in this endpoint's", text)

    def test_every_model_under_test_leaves_no_candidates(self):
        text = render_judge(self.inputs_with(("served",), model_name="served"),
                            {"model": "none"})
        self.assertIn("candidates: unknown", text)
        self.assertIn("Excluded", text)


class LineupCommand(unittest.TestCase):
    """The exact command, with every optional flag the inputs carry."""

    def test_optional_flags_appear_only_when_set(self):
        command = lineup_command("m", "http://h:1/v1", "", "hf-dir", "all", "", "", "runs/m")
        self.assertNotIn("--model-file", command.splitlines()[0])
        self.assertIn("--model-dir hf-dir", command.splitlines()[0])
        self.assertNotIn("--judge-model", command.splitlines()[0])
        self.assertNotIn("--judge-endpoint", command.splitlines()[0])

    def test_a_judge_endpoint_is_carried_through(self):
        command = lineup_command("m", "http://h:1/v1", "", "", "all", "j",
                                 "http://judge:9/v1", "runs/m")
        self.assertIn("--judge-model j", command.splitlines()[0])
        self.assertIn("--judge-endpoint http://judge:9/v1", command.splitlines()[0])


class KnowledgeTableEdges(unittest.TestCase):
    """A table that exists but holds no rows is as loud as one that is missing."""

    def test_a_table_without_rows_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.toml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# nothing here\n")
            with self.assertRaises(MissingKnowledge) as caught:
                load_knowledge(path, consult.ENGINE_TABLE_KEY)
        self.assertIn("no [[engine]] rows", str(caught.exception))


class ConsultCli(unittest.TestCase):
    """scripts/consult.py in process: the flags, the gathering, the exit codes."""

    def parser_defaults(self):
        from scripts.consult import build_parser
        return build_parser().parse_args([])

    def test_the_flags_have_their_defaults(self):
        args = self.parser_defaults()
        for flag in ("endpoint", "model_file", "model_dir", "engine", "use_case",
                     "judge_model", "out"):
            self.assertEqual(getattr(args, flag), "")
        self.assertEqual(args.api_key, "not-needed")

    def gather(self, argv, *, ids=("served-model",), fetch_error=None):
        """Run gather with the host, shape and endpoint probed."""
        from scripts.consult import build_parser, gather
        with mock.patch("scripts.consult.probe_host", return_value=HostFacts()), \
                mock.patch("scripts.consult.resolve_model_shape",
                           return_value=ModelShape(source=SOURCE_UNKNOWN)), \
                mock.patch("scripts.lib.consult.fetch_model_ids",
                           return_value=(ids, fetch_error)), \
                mock.patch("scripts.lib.consult.probe_engine_or_none",
                           return_value=SimpleNamespace(engine="llama.cpp")):
            return gather(build_parser().parse_args(argv))

    def test_without_an_endpoint_only_the_host_and_shape_are_gathered(self):
        inputs = self.gather([])
        self.assertEqual(inputs.endpoint, "")
        self.assertEqual(inputs.model_ids, ())
        self.assertIsNone(inputs.engine_facts)
        self.assertEqual(inputs.model_file, "")
        self.assertEqual(inputs.model_dir, "")
        self.assertEqual(inputs.engine_id, "")
        self.assertEqual(inputs.use_case_id, "")
        self.assertEqual(inputs.judge_pick, "")

    def test_a_live_endpoint_carries_the_ids_and_the_measured_engine(self):
        inputs = self.gather(["--endpoint", "http://h:9/v1", "--engine", "llama.cpp",
                              "--use-case", "ranking", "--judge-model", "deckard",
                              "--model-file", "m.gguf", "--model-dir", "hf"])
        self.assertEqual(inputs.endpoint, "http://h:9/v1")
        self.assertEqual(inputs.model_ids, ("served-model",))
        self.assertEqual(inputs.model_name, "served-model")
        self.assertEqual(inputs.engine_facts.engine, "llama.cpp")
        self.assertEqual(inputs.engine_id, "llama.cpp")
        self.assertEqual(inputs.use_case_id, "ranking")
        self.assertEqual(inputs.judge_pick, "deckard")
        self.assertEqual(inputs.model_file, "m.gguf")
        self.assertEqual(inputs.model_dir, "hf")

    def test_a_deaf_endpoint_falls_back_to_the_no_endpoint_sections(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            inputs = self.gather(["--endpoint", "http://h:9/v1"],
                                 fetch_error="connection refused")
        self.assertEqual(inputs.endpoint, "")
        self.assertEqual(inputs.model_ids, ())
        self.assertIn("note: the endpoint did not answer its model list", errors.getvalue())
        self.assertIn("connection refused", errors.getvalue())

    def main_with(self, argv, brief="stub brief"):
        from scripts.consult import build_parser, main
        with mock.patch("scripts.consult.build_parser", return_value=build_parser()), \
                mock.patch("scripts.lib.consult.render_brief", return_value=brief):
            buffer, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(errors):
                code = main(argv)
        return code, buffer.getvalue(), errors.getvalue()

    def test_main_prints_the_brief_and_exits_zero(self):
        code, out, _errors = self.main_with([])
        self.assertEqual(code, 0)
        self.assertEqual(out, "stub brief\n")

    def test_main_writes_the_brief_to_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "brief.md")
            code, _, _errors = self.main_with(["--out", path])
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "stub brief")

    def test_missing_knowledge_exits_two_and_names_the_gap(self):
        from scripts.consult import build_parser, main
        with mock.patch("scripts.consult.build_parser", return_value=build_parser()), \
                mock.patch("scripts.lib.consult.render_brief",
                           side_effect=MissingKnowledge("knowledge table x is missing")):
            errors = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                code = main([])
        self.assertEqual(code, 2)
        self.assertIn("consult: knowledge table x is missing", errors.getvalue())

    def test_the_script_form_prints_the_brief_and_exits_cleanly(self):
        """python3 scripts/consult.py must run: the guard is a real entry point."""
        argv = ["scripts/consult.py"]
        with mock.patch("sys.argv", argv), \
                mock.patch("scripts.lib.env_probe.probe_host", return_value=HostFacts()), \
                mock.patch("scripts.lib.model_probe.resolve_model_shape",
                           return_value=ModelShape(source=SOURCE_UNKNOWN)):
            buffer = io.StringIO()
            with redirect_stdout(buffer), self.assertRaises(SystemExit) as caught:
                runpy.run_path("scripts/consult.py", run_name="__main__")
        self.assertEqual(caught.exception.code, 0)
        brief = buffer.getvalue()
        for number in range(1, 9):
            self.assertIn(f"## {number}. ", brief)


if __name__ == "__main__":
    unittest.main()
