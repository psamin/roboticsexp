"""Ask for something; a Claude agent (Claude Agent SDK, your Claude Code login) plans it as simple pick-and-place
steps and Dream-1 does each one on the SO-101.

  uv run python agent.py                         # type requests; mock robot (no arm, no camera, no Dream credits)
  uv run python agent.py --voice                 # push-to-talk (Deepgram only while you talk); answers spoken
  uv run python agent.py --robot live --voice    # the real arm: every step asks y/N before Dream-1 moves it
  uv run python agent.py --once "hand me the medicine bottle"

The agent has three tools and nothing else (no shell, no files): look at the table, run one step, say something.
"""
import argparse
import asyncio
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock,
                              ToolUseBlock, create_sdk_mcp_server, tool)

import voice
from robot import MOCK_SCENE, LiveRobot, MockRobot

HERE = Path(__file__).resolve().parent
HANDOFF = "the front edge of the table"          # where "give it to me" puts things (nearest the user)

SYSTEM = f"""You are the planner for a single SO-101 robot arm on a table, talking with its owner.
Each step you run is carried out by Dream-1, a vision-language-action model that does ONE short pick-and-place
instruction at a time and does not plan. Your job:
1. Call look() to see the table from the overhead camera before planning, and again after each step to check it.
2. Break the request into the fewest simple steps. Write each step as one imperative sentence naming one visible
   object and one target, e.g. "Pick up the medicine bottle and place it on the plate". No pronouns, no
   multi-part steps, no reasoning in the step text.
3. "Give it to me", "bring me", "hand me" mean: place the object at {HANDOFF}.
4. Before the first step, say() the plan in one short sentence. Then run_step() each step in order.
5. If look() shows a step did not work, retry it once with the same wording; if it fails again, stop and say() so.
6. If an object you need is not visible or the request is unsafe or impossible for one small arm (heavy, liquids
   that could spill, knives, hot things, out of reach), do not move: say() why and ask what to do.
7. If a step is skipped by the user, do not repeat it; ask what they want instead.
Keep every say() under 20 words. Do not describe the image unless asked."""


def make_tools(robot, speak):
    @tool("look", "Photo of the table from the overhead camera.", {})
    async def look(args):
        return {"content": [{"type": "image", "data": await asyncio.to_thread(robot.look), "mimeType": "image/jpeg"}]}

    @tool("run_step", "Run one pick-and-place instruction on the arm with Dream-1. Blocks until it finishes.",
          {"step": str, "seconds": int})
    async def run_step(args):
        seconds = max(10, min(60, int(args.get("seconds") or 30)))
        return {"content": [{"type": "text", "text": await asyncio.to_thread(robot.run_step, args["step"], seconds)}]}

    @tool("say", "Say something short to the user.", {"text": str})
    async def say(args):
        print(f"robot: {args['text']}")
        await asyncio.to_thread(speak, args["text"])
        return {"content": [{"type": "text", "text": "said"}]}

    return [look, run_step, say]


async def run(args, robot, speak, listen):
    names = ("look", "run_step", "say")
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM, model=args.model, cwd=str(HERE), max_turns=40,
        mcp_servers={"robot": create_sdk_mcp_server("robot", tools=make_tools(robot, speak))},
        tools=[], allowed_tools=[f"mcp__robot__{n}" for n in names],
        setting_sources=[])                        # nothing from CLAUDE.md or project settings
    async with ClaudeSDKClient(options=options) as client:
        while True:
            text = args.once or await asyncio.to_thread(listen)
            if not text:
                continue
            print(f"you: {text}")
            if text.lower().strip(" .!") in ("quit", "exit", "goodbye"):
                break
            await client.query(text)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            print(f"  -> {block.name.split('__')[-1]} {block.input or ''}")
                        elif isinstance(block, TextBlock) and block.text.strip():
                            print(f"  (claude) {block.text.strip()}")
                elif isinstance(msg, ResultMessage) and msg.total_cost_usd:
                    print(f"  [{msg.num_turns} turns, ${msg.total_cost_usd:.3f} of Claude usage]")
            if args.once:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=("mock", "live"), default="mock")
    ap.add_argument("--voice", action="store_true", help="push-to-talk input and spoken answers")
    ap.add_argument("--once", help="run one request and exit")
    ap.add_argument("--scene", default=str(MOCK_SCENE), help="mock: the table image look() returns")
    ap.add_argument("--dream-args", default="", help="live: extra so101/dream_run.py flags, e.g. '--checkpoint ...'")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--quiet", action="store_true", help="print answers instead of speaking them")
    args = ap.parse_args()
    robot = MockRobot(args.scene) if args.robot == "mock" else LiveRobot(args.dream_args)
    speak = (lambda t: None) if args.quiet or not args.voice else voice.say
    listen = voice.listen if args.voice else (lambda: input("you: "))
    asyncio.run(run(args, robot, speak, listen))
    if robot.steps:
        print("steps run:", *robot.steps, sep="\n  ")


if __name__ == "__main__":
    main()
