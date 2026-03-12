"""
Customer-facing memory stack demo.

A practical onboarding script for:
- single-agent conversational memory flow (NIMMem0Manager)
- multi-agent handoff flow (MultiAgentMemory)
- full end-to-end mode (`--mode full`) for quick verification
"""

import argparse
import asyncio
import os
from typing import Dict, List, Optional, Tuple

GREEN = "\033[92m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _color(text: str, code: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{code}{text}{RESET}"


def _header(text: str, emoji: str = "•", enabled: bool = True) -> None:
    print()
    print(_color(f"{emoji} {text}", CYAN, enabled))
    print(_color("=" * (len(text) + 2), CYAN, enabled))


def _kv_row(label: str, value: str, enabled: bool = True) -> None:
    print(_color(f"{label}: ", GREEN, enabled) + value)


def _result_box(lines: List[str], enabled: bool = True) -> None:
    if not lines:
        return
    border = _color("┌" + "─" * 60 + "┐", MAGENTA, enabled)
    print(border)
    for line in lines:
        print(_color("│ " + line.ljust(58), MAGENTA, enabled) + _color(" │", MAGENTA, enabled))
    print(_color("└" + "─" * 60 + "┘", MAGENTA, enabled))


def _collect_turns(scenario: str) -> Tuple[List[str], int]:
    if scenario == "recovery":
        return [
            "Hi, I feel overloaded and hard to focus when I try to finish tasks.",
            "What coping structure did we agree on last time?",
            "Can you give me a short 3-step reset?",
        ], 3
    if scenario == "goals":
        return [
            "Hi, I prefer practical, direct coaching.",
            "What should I prioritize this week?",
            "Can we recap what I said earlier?",
        ], 3
    # default: coaching
    return [
        "Hi, I prefer direct but supportive coaching.",
        "What was I trying to work on in my last conversation?",
    ], 2


def _prompt_select(prompt: str, options: List[str], default: int = 1) -> int:
    while True:
        print(_color(prompt, CYAN))
        for idx, option in enumerate(options, start=1):
            print(_color(f"  {idx}) {option}", GREEN))
        choice = input(_color(f"Select [1-{len(options)}] (default {default}): ", MAGENTA)).strip()
        if not choice:
            return default
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(options):
                return selected
        print(_color("Invalid selection. Please pick a valid number.", RED))


def _format_turn_result(turn: int, prompt: str, result: Dict[str, object], enabled: bool) -> None:
    print(_color(f"\nTurn {turn}", MAGENTA, enabled))
    print(_color(f" User: ", GREEN, enabled) + prompt)
    print(_color(f" Bot : ", GREEN, enabled) + str(result["response"]))
    _kv_row(
        " Trace",
        (
            f"request={result['request_id']} | "
            f"memories={result['memories_used']} | "
            f"latency={result['latency_ms']}ms | "
            f"crisis={result['crisis_flagged'] if 'crisis_flagged' in result else False}"
        ),
        enabled=enabled,
    )


from ai.core.memory.mem0_nim import (
    AgentIdentity,
    AgentRole,
    MemoryScope,
    MultiAgentMemory,
    NIMMem0Config,
    NIMMem0Manager,
    create_empathy_gym_context,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Pixelated memory quickstart demo with rich usage guidance."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "full"],
        default="single",
        help=(
            "single = NIMMem0Manager flow, "
            "multi = MultiAgentMemory handoff flow, "
            "full = both flows in one run"
        ),
    )
    parser.add_argument("--user", default=os.getenv("MEM0_USER_ID", "demo_user"))
    parser.add_argument("--session", default=os.getenv("MEM0_SESSION_ID", "demo_session"))
    parser.add_argument("--nim-key", default=os.getenv("NIM_API_KEY"))
    parser.add_argument("--mem0-key", default=os.getenv("MEM0_API_KEY"))
    parser.add_argument(
        "--scenario",
        choices=["coaching", "goals", "recovery"],
        default="coaching",
        help="Conversation personality for single-agent mode.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output for logs.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch guided menu mode to pick a flow at runtime.",
    )
    return parser.parse_args()


async def run_single_mode(user_id: str, nim_key: Optional[str], scenario: str = "coaching", use_color: bool = True):
    turns, _ = _collect_turns(scenario)

    if not nim_key:
        raise SystemExit(
            "Missing NIM API key. Set --nim-key or NIM_API_KEY to run the single-user demo."
        )

    _header("Single-Agent Flow", "🚀", enabled=use_color)
    manager = NIMMem0Manager(
        NIMMem0Config(
            nim_api_key=nim_key,
            user_id=user_id,
        )
    )

    for idx, turn in enumerate(turns, start=1):
        result = await manager.get_response(turn)
        _format_turn_result(idx, turn, result, enabled=use_color)
        await asyncio.sleep(0.05)


async def run_multi_mode(user_id: str, session_id: str, mem0_key: Optional[str], use_color: bool = True):
    if not mem0_key:
        raise SystemExit(
            "Missing Mem0 API key. Set --mem0-key or MEM0_API_KEY to run the multi-agent demo."
        )

    _header("Multi-Agent Flow", "🧠", enabled=use_color)
    memory = MultiAgentMemory(api_key=mem0_key)
    context = create_empathy_gym_context(user_id=user_id, session_id=session_id, current_role=AgentRole.TRAINER)

    _kv_row("Session", f"{user_id}/{session_id}", enabled=use_color)
    _kv_row(
        "Agents",
        ", ".join([agent.name for agent in context.agents]),
        enabled=use_color,
    )

    await memory.store_agent_memory(
        context,
        "Session kickoff: user prefers guided coaching with clear steps.",
        scope=MemoryScope.SHARED,
    )
    shared = await memory.get_shared_context(context, limit=5)
    _kv_row(f"Shared memories visible", str(len(shared)), enabled=use_color)
    if shared:
        _result_box([item.get("memory", "") or item.get("content", "") for item in shared[:3]], use_color)

    feedback_agent = AgentIdentity(
        agent_id=f"feedback_{session_id}",
        role=AgentRole.FEEDBACK,
        name="Feedback Agent",
    )
    result = await memory.handoff_to_agent(context, feedback_agent, "Switching to scoring + feedback phase")
    _kv_row(
        "Handoff success",
        _color(str(result["success"]), GREEN if result["success"] else RED, use_color),
        enabled=use_color,
    )
    if result["success"]:
        _kv_row("Handoff memory id", result["handoff_memory_id"], enabled=use_color)
        handoff = result["handoff"]
        _kv_row("Transferred memories", str(handoff["transferred"]), enabled=use_color)
        _kv_row(
            "Summary",
            f"{handoff['source_agent']} -> {handoff['target_agent']}",
            enabled=use_color,
        )


async def run_full_mode(args):
    _header("Full Stack Verification", "✨", enabled=not args.no_color)
    _kv_row("Mode", "single + multi", enabled=not args.no_color)
    await run_single_mode(args.user, args.nim_key, args.scenario, use_color=not args.no_color)
    await run_multi_mode(
        args.user, args.session, args.mem0_key, use_color=not args.no_color
    )


async def run_interactive_mode(args):
    _header("Interactive Memory Stack Demo", "🎛️", enabled=not args.no_color)
    selected_mode = _prompt_select(
        "Pick a flow to run:",
        [
            "single  (NIMMem0Manager chatbot + continuity)",
            "multi   (MultiAgentMemory handoff flow)",
            "full    (single + multi)",
        ],
    )
    scenario = "coaching"
    if selected_mode == 1:
        selected_scenario = _prompt_select(
            "Pick your conversation tone:",
            [
                "coaching (direct, supportive)",
                "goals (task-focused)",
                "recovery (grounded, emotional support)",
            ],
        )
        scenario = ["coaching", "goals", "recovery"][selected_scenario - 1]
        user = input(_color("Enter user id [demo_user]: ", CYAN)).strip() or "demo_user"
        nim_key = os.getenv("NIM_API_KEY") or args.nim_key
        if not nim_key:
            nim_key = input(_color("Enter NIM API key: ", CYAN)).strip() or None
        await run_single_mode(user, nim_key, scenario=scenario, use_color=not args.no_color)
    elif selected_mode == 2:
        user = input(_color("Enter user id [demo_user]: ", CYAN)).strip() or "demo_user"
        session = input(_color("Enter session id [demo_session]: ", CYAN)).strip() or "demo_session"
        mem0_key = os.getenv("MEM0_API_KEY") or args.mem0_key
        if not mem0_key:
            mem0_key = input(_color("Enter Mem0 API key: ", CYAN)).strip() or None
        await run_multi_mode(user, session, mem0_key, use_color=not args.no_color)
    else:
        user = input(_color("Enter user id [demo_user]: ", CYAN)).strip() or "demo_user"
        session = input(_color("Enter session id [demo_session]: ", CYAN)).strip() or "demo_session"
        selected_scenario = _prompt_select(
            "Pick your conversation tone:",
            [
                "coaching (direct, supportive)",
                "goals (task-focused)",
                "recovery (grounded, emotional support)",
            ],
        )
        scenario = ["coaching", "goals", "recovery"][selected_scenario - 1]
        nim_key = os.getenv("NIM_API_KEY") or args.nim_key
        if not nim_key:
            nim_key = input(_color("Enter NIM API key: ", CYAN)).strip() or None
        mem0_key = os.getenv("MEM0_API_KEY") or args.mem0_key
        if not mem0_key:
            mem0_key = input(_color("Enter Mem0 API key: ", CYAN)).strip() or None
        full_args = argparse.Namespace(
            user=user,
            session=session,
            scenario=scenario,
            nim_key=nim_key,
            mem0_key=mem0_key,
            no_color=args.no_color,
        )
        await run_full_mode(full_args)


def main():
    args = parse_args()
    print(
        _color(
            "Pixelated Memory Stack Demo",
            GREEN,
            enabled=not args.no_color,
        )
    )
    print(_color("================================", YELLOW, enabled=not args.no_color))

    if args.interactive:
        asyncio.run(run_interactive_mode(args))
    elif args.mode == "single":
        asyncio.run(run_single_mode(args.user, args.nim_key, args.scenario, use_color=not args.no_color))
    elif args.mode == "multi":
        asyncio.run(run_multi_mode(args.user, args.session, args.mem0_key, use_color=not args.no_color))
    elif args.mode == "full":
        asyncio.run(run_full_mode(args))


if __name__ == "__main__":
    main()
