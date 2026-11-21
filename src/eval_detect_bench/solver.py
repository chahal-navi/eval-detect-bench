"""Append the probe question to the transcript, ported from the reference `inject_pq`."""

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolInfo

from eval_detect_bench.dataset import pool_for_model
from eval_detect_bench.probes import load_probes

# Stands in for a missing assistant turn on Claude judges (see `prepare_messages`).
CLAUDE_PLACEHOLDER = "<placeholder>"


def close_dangling_tool_calls(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Give every unanswered tool call an empty result.

    Some transcripts end on an assistant tool call, and a few have an assistant
    turn where only some calls got results. Provider APIs such as Anthropic's
    reject both. An empty result is inserted after each such turn's existing
    results. The reference only handled a trailing assistant turn; the partial
    case matches the prompt in the authors' published Haiku log
    (dataclaw:67e481b7-..., which gained an empty result for its 7th call).
    """
    out: list[ChatMessage] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        out.append(message)
        i += 1
        if not (isinstance(message, ChatMessageAssistant) and message.tool_calls):
            continue
        answered: set[str] = set()
        while i < len(messages) and isinstance(messages[i], ChatMessageTool):
            tool_message = messages[i]
            assert isinstance(tool_message, ChatMessageTool)
            if tool_message.tool_call_id is not None:
                answered.add(tool_message.tool_call_id)
            out.append(tool_message)
            i += 1
        out.extend(
            ChatMessageTool(content="", tool_call_id=tc.id, function=tc.function)
            for tc in message.tool_calls
            if tc.id not in answered
        )
    return out


def prepare_messages(
    messages: list[ChatMessage], model_name: str, claude_placeholder: bool | None = None
) -> list[ChatMessage]:
    """Apply the reference's Claude-specific fix-up before the probe is appended.

    The fix-up drops a trailing system message and, if the transcript ends on a
    user message, adds a placeholder assistant turn before the probe.

    By default (`claude_placeholder=None`) it follows the reference, which checks
    `model.api.model_name.startswith("claude")`. That is true for
    `anthropic/claude-*` but not for `openrouter/anthropic/claude-*`, whose model
    name is `anthropic/claude-*`. The paper ran Haiku 4.5 directly and Sonnet and
    Opus 4.6 through OpenRouter, so only Haiku got the fix-up. `True` or `False`
    forces it on or off, e.g. to reproduce the paper's Haiku prompts through
    OpenRouter.
    """
    messages = list(messages)
    apply = model_name.startswith("claude") if claude_placeholder is None else claude_placeholder
    if apply:
        if messages and isinstance(messages[-1], ChatMessageSystem):
            messages.pop()
        if messages and isinstance(messages[-1], ChatMessageUser):
            messages.append(ChatMessageAssistant(content=CLAUDE_PLACEHOLDER))
    return close_dangling_tool_calls(messages)


def _never_called() -> Any:
    """Make a fresh placeholder function for one tool.

    Each tool needs its own function object: Inspect records the tool's name on
    the function when it is attached, so tools sharing one function would all
    end up with the last tool's name.
    """

    async def never_called() -> str:
        return ""

    return never_called


def tool_defs(tools: list[dict[str, Any]]) -> list[ToolDef]:
    """Rebuild the transcript's tool schemas so the judge sees them, uncallable.

    Parameters with no description get their name as a description, as in the
    reference: some providers reject parameters without one.
    """
    defs: list[ToolDef] = []
    for raw in tools:
        info = ToolInfo.model_validate(raw)
        for name, param in info.parameters.properties.items():
            param.description = param.description or name
        defs.append(
            ToolDef(
                tool=_never_called(),
                name=info.name,
                description=info.description,
                parameters=info.parameters,
            )
        )
    return defs


@solver
def inject_probe(claude_placeholder: bool | None = None, judge_pool: str | None = None) -> Solver:
    """Append the sample's probe question after the transcript.

    The transcript's tools are attached with `tool_choice="none"`, so the judge
    can only answer in text. Inspect passes the tool schemas only to providers
    that require them (Anthropic, once tool messages are present).

    Args:
        claude_placeholder: See `prepare_messages`. None follows the reference.
        judge_pool: The pool chosen from the judge model when the task was
            built. If set, the model actually judging must map to the same pool;
            this catches a task built for one model and run with another.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        if judge_pool is not None and pool_for_model(str(model)) != judge_pool:
            raise RuntimeError(
                f"The task picked the '{judge_pool}' transcript pool for its judge, but "
                f"'{model}' is judging. Pass -T transcripts=<pool> to judge a pool "
                "with a different model."
            )
        probe = load_probes()[state.metadata["probe_id"]]
        model_name = model.api.model_name
        messages = prepare_messages(state.messages, model_name, claude_placeholder)
        state.messages = [*messages, ChatMessageUser(content=probe.template_text)]
        state.tools = tool_defs(state.metadata.get("tools", []))  # type: ignore[assignment]
        state.tool_choice = "none"
        return state

    return solve
