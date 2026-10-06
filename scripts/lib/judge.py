"""Prose ELO judge — an LLM that compares two model outputs."""
from .model import ModelConfig, chat_request

JUDGE_SYSTEM = """You are an impartial judge evaluating text quality. Compare the two responses below.
Rate which one is better for the given task. Consider: clarity, completeness, adherence to the prompt,
tone, structure, and overall effectiveness.

You must respond with exactly one word: "A" if Response A is better, "B" if Response B is better,
or "TIE" if they are equally good or both fail at the task."""


def setup_judge(endpoint: str = "http://localhost:11434/v1",
                model: str = "hermes-4-70b",
                api_key: str = "not-needed") -> ModelConfig:
    """Configure the judge model. Should be a capable instruction-following model."""
    return ModelConfig(name=model, endpoint=endpoint, api_key=api_key,
                       temperature=0.0, token_limit=8192)


def judge_pair(cfg: ModelConfig, task: str, response_a: str, response_b: str) -> str:
    """Compare two responses. Returns 'A', 'B', or 'TIE'."""
    prompt = f"""## Task
{task}

## Response A
{response_a}

## Response B
{response_b}

## Decision
Which response is better? Answer A, B, or TIE."""
    result = chat_request(cfg, [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ], max_tokens=16, temperature=0.0)
    if result is None:
        return "TIE"
    result = result.strip().upper()
    if "A" in result and "B" not in result:
        return "A"
    if "B" in result and "A" not in result:
        return "B"
    return "TIE"