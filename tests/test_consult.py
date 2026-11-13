"""The consult brief: sections present, tables read by id, unknowns stated.

The brief is rendered from facts, so the assertions are on stable things -- the eight
section headings, a row's own serve text read back from engines.toml, a judge candidate
that must be excluded -- never on wording the brief happens to use. HTTP is stubbed the
same way the rest of the suite stubs it: a local JsonStub, no external calls.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib import consult  # noqa: E402
from scripts.lib.consult import (  # noqa: E402
    BriefInputs,
    MissingKnowledge,
    SMOKE_CONCRETE,
    base_url,
    detect_model_types,
    engine_candidates,
    engine_row,
    fetch_model_ids,
    judge_candidates,
    lineup_command,
    load_knowledge,
    render_brief,
    rows_by_id,
)
from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import Gpu, HostFacts, ToolFact  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.model_probe import (  # noqa: E402
    SOURCE_GGUF,
    SOURCE_HF,
    SOURCE_UNKNOWN,
    ModelShape,
)
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


if __name__ == "__main__":
    unittest.main()
