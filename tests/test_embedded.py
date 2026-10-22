"""ouroboros.embedded: the bare ReAct loop a host harness plugs in (no SYSTEM, BIBLE, memory or built-in tools)."""

from __future__ import annotations

import json

import pytest

from ouroboros.embedded import EmbeddedTool, run_react


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, model, tools=None, **kwargs):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools, "model": model, **kwargs})
        return self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 3}


def _call(name, args, raw=False):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c-" + name, "type": "function", "function": {"name": name, "arguments": args if raw else json.dumps(args)}}]}


def test_bare_loop_runs_host_tools_and_finishes():
    seen = []
    lookup = EmbeddedTool("lookup", "read a field", {"type": "object", "properties": {"field": {"type": "string"}}},
                          lambda a: seen.append(a) or {"value": "Работает"})
    llm = ScriptedLLM([_call("lookup", {"field": "status"}), _call("submit_result", {"decision": "yes"})])
    res = run_react([{"role": "user", "content": "step S10"}], [lookup], model="openai-compatible::m", llm=llm)
    assert res.stop_reason == "finish_tool" and res.finish_args == {"decision": "yes"}
    assert seen == [{"field": "status"}] and res.rounds == 2 and res.tokens_in == 20
    first = llm.seen[0]
    # only host messages and host tools: no SYSTEM.md / BIBLE.md injected, no built-in Ouroboros tools
    assert first["messages"] == [{"role": "user", "content": "step S10"}]
    assert [t["function"]["name"] for t in first["tools"]] == ["lookup"]
    assert llm.seen[1]["messages"][-1]["role"] == "tool" and "Работает" in llm.seen[1]["messages"][-1]["content"]


def test_double_encoded_args_unknown_tool_and_final_text():
    llm = ScriptedLLM([_call("nope", "{}", raw=True), _call("submit_result", json.dumps(json.dumps({"decision": "no"})), raw=True)])
    res = run_react([{"role": "user", "content": "x"}], [], model="m", llm=llm)
    assert res.finish_args == {"decision": "no"} and res.tool_calls[0]["ok"] is False
    llm = ScriptedLLM([{"role": "assistant", "content": "готово"}])
    assert run_react([{"role": "user", "content": "x"}], [], model="m", llm=llm).stop_reason == "final_text"


def test_host_signal_passes_through_and_limits():
    class Handover(Exception):
        pass

    def boom(_a):
        raise Handover("to a human")

    tool = EmbeddedTool("write", "write", {"type": "object"}, boom)
    with pytest.raises(Handover):
        run_react([{"role": "user", "content": "x"}], [tool], model="m", llm=ScriptedLLM([_call("write", {})]), passthrough=(Handover,))
    many = ScriptedLLM([_call("write", {})] * 3)
    res = run_react([{"role": "user", "content": "x"}], [EmbeddedTool("write", "w", {"type": "object"}, lambda a: "ok")], model="m", llm=many, max_rounds=3)
    assert res.stop_reason == "max_rounds" and len(res.tool_calls) == 3
    with pytest.raises(ValueError):
        run_react([], [], model="m", llm=many, profile="full")
