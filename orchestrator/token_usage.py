"""Summarize agent token usage from one run's isolated log directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "cost_usd",
)


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _normalized(raw: Any) -> dict[str, int | float | None] | None:
    if not isinstance(raw, dict):
        return None
    input_tokens = _integer(raw.get("input_tokens"))
    cache_creation_input_tokens = _integer(raw.get("cache_creation_input_tokens"))
    cached_input_tokens = _integer(raw.get("cached_input_tokens"))
    output_tokens = _integer(raw.get("output_tokens"))
    reasoning_output_tokens = _integer(raw.get("reasoning_output_tokens"))
    total_tokens = _integer(raw.get("total_tokens"))
    cost_usd = _number(raw.get("cost_usd"))

    input_details = raw.get("input_tokens_details")
    if cached_input_tokens is None and isinstance(input_details, dict):
        cached_input_tokens = _integer(input_details.get("cached_tokens"))
    output_details = raw.get("output_tokens_details")
    if reasoning_output_tokens is None and isinstance(output_details, dict):
        reasoning_output_tokens = _integer(output_details.get("reasoning_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        # Cached input is a subset of input_tokens, not an additional charge.
        total_tokens = input_tokens + output_tokens

    values = {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }
    return values if any(value is not None for value in values.values()) else None


def _json_lines(path: Path):
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                yield payload


def _session_usage(
    path: Path,
) -> tuple[str | None, dict[str, int | float | None] | None, list[dict[str, Any]]]:
    session_id: str | None = None
    latest: dict[str, int | float | None] | None = None
    turns: list[dict[str, Any]] = []
    for payload in _json_lines(path):
        if payload.get("type") == "session_meta":
            candidate = payload.get("payload", {}).get("id")
            if isinstance(candidate, str):
                session_id = candidate
        if payload.get("type") != "event_msg":
            continue
        event = payload.get("payload")
        if not isinstance(event, dict) or event.get("type") != "token_count":
            continue
        info = event.get("info")
        if not isinstance(info, dict):
            continue
        candidate = _normalized(info.get("total_token_usage"))
        if candidate is not None:
            latest = candidate
        per_turn = _normalized(info.get("last_token_usage"))
        if per_turn is not None:
            turns.append(
                {
                    "sequence": len(turns) + 1,
                    "timestamp": payload.get("timestamp"),
                    **per_turn,
                }
            )
    return session_id, latest, turns


def _attempt_totals(
    paths: list[Path],
) -> tuple[dict[str, int | float | None] | None, list[dict[str, Any]]]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    known = {field: True for field in TOKEN_FIELDS}
    turns: list[dict[str, Any]] = []
    for path in paths:
        for payload in _json_lines(path):
            if payload.get("type") != "turn.completed":
                continue
            usage = _normalized(payload.get("usage"))
            if usage is None:
                continue
            turns.append(
                {
                    "sequence": len(turns) + 1,
                    "attempt_log": path.name,
                    "timestamp": payload.get("timestamp"),
                    **usage,
                }
            )
            for field in TOKEN_FIELDS:
                value = usage[field]
                if value is None:
                    known[field] = False
                else:
                    totals[field] += value
    if not turns:
        return None, []
    return {
        field: totals[field] if known[field] else None
        for field in TOKEN_FIELDS
    }, turns


def _claude_model_usage(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for model, value in raw.items():
        if not isinstance(model, str) or not isinstance(value, dict):
            continue
        normalized[model] = {
            "input_tokens": _integer(value.get("inputTokens")),
            "cache_creation_input_tokens": _integer(value.get("cacheCreationInputTokens")),
            "cached_input_tokens": _integer(value.get("cacheReadInputTokens")),
            "output_tokens": _integer(value.get("outputTokens")),
            "cost_usd": _number(value.get("costUSD")),
            "canonical_model": value.get("canonicalModel"),
            "provider": value.get("provider"),
            "context_window": _integer(value.get("contextWindow")),
            "max_output_tokens": _integer(value.get("maxOutputTokens")),
        }
    return normalized


def _merge_model_usage(
    aggregate: dict[str, dict[str, Any]], incoming: dict[str, dict[str, Any]]
) -> None:
    numeric = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "cost_usd",
    )
    for model, values in incoming.items():
        target = aggregate.setdefault(model, {})
        for field in numeric:
            value = values.get(field)
            if value is not None:
                previous = target.get(field)
                target[field] = (
                    previous if isinstance(previous, (int, float)) else 0
                ) + value
            else:
                target.setdefault(field, None)
        for field in ("canonical_model", "provider", "context_window", "max_output_tokens"):
            if values.get(field) is not None:
                target[field] = values[field]


def _summarize_claude(
    agent_log_dir: Path, session_id: str | None
) -> dict[str, Any]:
    attempt_logs = sorted(agent_log_dir.glob("attempt-*.jsonl"))
    totals: dict[str, int | float] = {field: 0 for field in TOKEN_FIELDS}
    known = {field: True for field in TOKEN_FIELDS}
    turns: list[dict[str, Any]] = []
    model_usage: dict[str, dict[str, Any]] = {}
    for path in attempt_logs:
        result: dict[str, Any] | None = None
        for payload in _json_lines(path):
            if payload.get("type") == "result":
                result = payload
        if result is None:
            continue
        recorded_session = result.get("session_id")
        if session_id and recorded_session and recorded_session != session_id:
            continue
        usage = result.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens = _integer(usage.get("input_tokens"))
        cache_creation = _integer(usage.get("cache_creation_input_tokens"))
        cached = _integer(usage.get("cache_read_input_tokens"))
        output = _integer(usage.get("output_tokens"))
        output_details = usage.get("output_tokens_details")
        reasoning = None
        if isinstance(output_details, dict):
            reasoning = _integer(
                output_details.get("thinking_tokens", output_details.get("reasoning_tokens"))
            )
        if reasoning is None:
            reasoning = 0
        per_model = _claude_model_usage(
            result.get("modelUsage", result.get("model_usage"))
        )
        cost = _number(result.get("total_cost_usd"))
        if cost is None and per_model:
            model_costs = [value.get("cost_usd") for value in per_model.values()]
            if all(value is not None for value in model_costs):
                cost = sum(model_costs)
        total = None
        if None not in (input_tokens, cache_creation, cached, output):
            # Anthropic reports uncached, cache-write and cache-read input as disjoint
            # categories, unlike Codex where cached input is a subset of input.
            total = input_tokens + cache_creation + cached + output
        values: dict[str, int | float | None] = {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cached_input_tokens": cached,
            "output_tokens": output,
            "reasoning_output_tokens": reasoning,
            "total_tokens": total,
            "cost_usd": cost,
        }
        turns.append(
            {
                "sequence": len(turns) + 1,
                "attempt_log": path.name,
                "session_id": recorded_session,
                "is_error": bool(result.get("is_error")),
                "models": sorted(per_model),
                **values,
            }
        )
        for field, value in values.items():
            if value is None:
                known[field] = False
            else:
                totals[field] += value
        _merge_model_usage(model_usage, per_model)
    fields = {
        field: totals[field] if turns and known[field] else None
        for field in TOKEN_FIELDS
    }
    populated = sum(fields[field] is not None for field in TOKEN_FIELDS)
    status = "complete" if populated == len(TOKEN_FIELDS) else "partial" if populated else "missing"
    return {
        "schema_version": 3,
        "agent": "claude",
        "status": status,
        "source": "claude_attempt_result" if turns else "none",
        # This is whatever the agent CLI reported. It is provenance, not a guarantee
        # that a third-party gateway uses the same billing schedule.
        "cost_source": "agent_cli_reported" if fields["cost_usd"] is not None else None,
        "cost_is_official_billing": False,
        "session_id": session_id,
        "session_jsonl": None,
        "attempt_logs": [item.name for item in attempt_logs],
        "completed_attempts_counted": len(turns),
        "turn_count": len(turns),
        "turns": turns,
        "model_usage": model_usage,
        **fields,
    }


def summarize_token_usage(
    agent_log_dir: Path,
    *,
    session_id: str | None = None,
    agent: str = "codex",
) -> dict[str, Any]:
    """Return one non-ambiguous token receipt for a Codex run.

    A Codex session emits cumulative totals after each model response. The last
    cumulative snapshot is authoritative and includes resumed attempts. Older CLI
    versions may expose usage only on ``turn.completed`` in the host attempt logs;
    those per-turn values are summed as a documented fallback.
    """

    agent_log_dir = agent_log_dir.resolve()
    if agent == "claude":
        return _summarize_claude(agent_log_dir, session_id)
    if agent != "codex":
        raise ValueError(f"unknown agent for token receipt: {agent}")
    session_files = sorted(agent_log_dir.glob("*-home/sessions/**/*.jsonl"))
    candidates: list[tuple[Path, dict[str, int | float | None], list[dict[str, Any]]]] = []
    for path in session_files:
        recorded_session, usage, turns = _session_usage(path)
        if usage is None:
            continue
        if session_id and recorded_session and recorded_session != session_id:
            continue
        candidates.append((path, usage, turns))

    attempt_logs = sorted(agent_log_dir.glob("attempt-*.jsonl"))
    if candidates:
        # An isolated CODEX_HOME normally contains one session. If a stale file is
        # present, the largest cumulative total is the least lossy terminal snapshot.
        path, usage, turns = max(
            candidates, key=lambda item: item[1].get("total_tokens") or -1
        )
        source = "session_jsonl_total_token_usage"
        completed_attempts = None
    else:
        usage, turns = _attempt_totals(attempt_logs)
        completed_attempts = len(turns)
        path = None
        source = "attempt_turn_completed" if usage is not None else "none"

    fields = usage or {field: None for field in TOKEN_FIELDS}
    required = (
        "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_output_tokens", "total_tokens",
    )
    populated = sum(fields[field] is not None for field in required)
    status = "complete" if populated == len(required) else "partial" if populated else "missing"
    return {
        "schema_version": 3,
        "agent": "codex",
        "status": status,
        "source": source,
        "cost_source": "agent_trace_reported" if fields["cost_usd"] is not None else None,
        "cost_is_official_billing": False,
        "session_id": session_id,
        "session_jsonl": str(path.relative_to(agent_log_dir)) if path else None,
        "attempt_logs": [item.name for item in attempt_logs],
        "completed_attempts_counted": completed_attempts,
        "turn_count": len(turns),
        "turns": turns,
        "model_usage": {},
        **fields,
    }


def write_token_usage(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
