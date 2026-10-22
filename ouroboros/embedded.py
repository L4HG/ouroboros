"""Embedded ReAct loop for host harnesses: Ouroboros as a pluggable engine.

A host (for example the bank execution harness in ``L4HG/ops_toolhub``) already owns the task graph, the
policies, the journal and the tools. It needs only a tool-calling loop over a model. ``run_react`` gives it
exactly that, built from Ouroboros parts that matter for weak and regional models:

* ``LLMClient`` provider routing: ``openai-compatible::<model>`` (any OpenAI-style endpoint, base URL from
  ``OPENAI_COMPATIBLE_BASE_URL``), ``gigachat::<model>`` (native GigaChat functions), ``deepseek::…`` and the rest;
* the OpenAI function-tool format Ouroboros already speaks.

Profile ``bare`` (the only embedded profile) deliberately leaves out everything that makes Ouroboros an agent with
a self: no SYSTEM.md, no BIBLE.md, no memory or knowledge sections, no built-in tools, no safety/review LLM calls,
no consciousness, no self-modification, no supervisor or server. The host passes its own messages and tools; the
loop ends when the model calls ``finish_tool`` (its arguments are the result), answers without a tool call, or
hits ``max_rounds`` / ``max_total_tokens``.

The full Ouroboros (constitution, memory, review, evolution) stays available the usual way — ``ouroboros.cli``
server plus ``run`` — and can reach host tools over MCP; it is not part of this module on purpose: a bank step
executor must not inherit a constitution it did not sign up for.

    from ouroboros.embedded import EmbeddedTool, run_react
    res = run_react(messages, [EmbeddedTool("submit_result", "…", schema, handler)], model="openai-compatible::Qwen2.5-72B")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

PROFILES = ("bare",)


@dataclass
class EmbeddedTool:
    """A host tool: ``handler(args) -> str | dict``; exceptions become an ``{"error": …}`` tool result."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]

    def schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class EmbeddedResult:
    stop_reason: str                 # finish_tool | final_text | max_rounds | max_tokens | llm_error
    final_text: str = ""
    finish_args: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        if isinstance(value, str):  # some providers double-encode arguments
            value = json.loads(value)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def run_react(
    messages: List[Dict[str, Any]],
    tools: List[EmbeddedTool],
    *,
    model: str,
    profile: str = "bare",
    finish_tool: str = "submit_result",
    max_rounds: int = 8,
    max_tokens: int = 2048,
    max_total_tokens: int = 0,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    llm: Any = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    passthrough: tuple = (),
) -> EmbeddedResult:
    """Plain ReAct over host messages and tools. ``passthrough`` — exception types to re-raise from tool handlers
    (host policy signals such as «hand the task to a human»); every other handler exception is shown to the model."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r}: embedded mode supports {PROFILES}; full Ouroboros runs via ouroboros.cli")
    if llm is None:
        from ouroboros.llm import LLMClient
        llm = LLMClient()
    emit = on_event or (lambda _kind, _data: None)
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    convo: List[Dict[str, Any]] = [dict(m) for m in messages]
    res = EmbeddedResult(stop_reason="max_rounds")
    for round_idx in range(max_rounds):
        res.rounds = round_idx + 1
        if max_total_tokens and res.tokens_in + res.tokens_out > max_total_tokens:
            res.stop_reason = "max_tokens"
            return res
        try:
            msg, usage = llm.chat(convo, model=model, tools=schemas or None, reasoning_effort=reasoning_effort,
                                  max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 - the host decides what an LLM outage means for its task
            res.stop_reason, res.error = "llm_error", f"{type(exc).__name__}: {exc}"
            return res
        res.tokens_in += int((usage or {}).get("prompt_tokens") or 0)
        res.tokens_out += int((usage or {}).get("completion_tokens") or 0)
        emit("llm", {"round": round_idx, "usage": {"in": res.tokens_in, "out": res.tokens_out}})
        calls = (msg or {}).get("tool_calls") or []
        if not calls:
            res.stop_reason, res.final_text = "final_text", (msg or {}).get("content") or ""
            return res
        call = calls[0]  # one call per round: weak models lose track of parallel calls
        fn = call.get("function") or {}
        name, args = fn.get("name", ""), _parse_args(fn.get("arguments"))
        convo.append({"role": "assistant", "content": (msg or {}).get("content") or "", "tool_calls": [call]})
        if name == finish_tool and finish_tool not in by_name:
            res.stop_reason, res.finish_args = "finish_tool", args
            return res
        tool = by_name.get(name)
        if tool is None:
            out: Any = {"error": f"unknown tool {name!r}; available: {', '.join(by_name)}"}
        else:
            try:
                out = tool.handler(args)
            except passthrough:
                raise
            except Exception as exc:  # noqa: BLE001 - the model sees the failure and decides
                out = {"error": f"{type(exc).__name__}: {exc}"}
        res.tool_calls.append({"tool": name, "args": args, "ok": not (isinstance(out, dict) and "error" in out)})
        emit("tool", {"tool": name})
        if name == finish_tool and not (isinstance(out, dict) and "error" in out):
            res.stop_reason, res.finish_args = "finish_tool", args
            return res
        convo.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{round_idx}",
                      "content": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)})
    return res
