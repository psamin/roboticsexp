# Voice "fetch it" agent on the SO-101: what exists, what to build

Survey of 2024–2026 work, as of 2026-11-05. **V** = source opened (by me or a research subagent). **U** = unverified (search snippet, memory, or my estimate). **D** = my arithmetic, with the assumptions given.

## Answer

Build architecture **(b)** first:

- **Planner.** Claude in the Agent SDK. It calls a handful of coarse robot tools.
- **Perception.** Open-vocabulary detection on the overhead camera, which is already calibrated.
- **Motion.** Back-project the object to the table, run IK, then do a scripted grasp and move to a fixed handover pose.
- **Voice.** Push-to-talk, local STT (parakeet-mlx), and `say` for speech output.
- **Learned policies** (SmolVLA, Dream-1, MolmoAct2) become an optional backend for the `pick` tool, compared against the scripted grasp on the same trials.

Why:

- No one has published real-SO-101 success rates for voice or LLM multi-step fetching. The builds that do work ground the object with a VLM or detector and move with IK (§1).
- **Learned policies are weak on SO-101 so far.**
  - Fine-tuned VLAs on SO-101 average 32–56% per task with 100 demos per task.
  - They score **0–10% on the one language-conditioned task** ([Yu & Qiu 2026](https://arxiv.org/html/2606.08881v2), V).
  - Your own runs agree: Dream-1 zero-shot closed on an object (held ~5 s) in 1 of the 17 runs tabled in the log, and the SmolVLA fine-tune is 0/1 ([DREAM_DEBUG_LOG](../so101/DREAM_DEBUG_LOG.md), [so101 README](../so101/README.md)).
- (b) needs no new demos. Its hardest prerequisite, camera-to-base calibration (K, R, t at 1.29 px reprojection error), is already done in [camera_calib.json](../so101-bounce/real/camera_calib.json).

---

## 1. Existing systems

### 1a. LLM and voice agents on SO-100/101

| System | What it does | Results | Effort to adopt | Status |
|---|---|---|---|---|
| [IliaLarchenko/robot_MCP](https://github.com/IliaLarchenko/robot_MCP) | MCP server for SO-100/101. Tools: `get_robot_state` (camera images), `move_robot` (relative mm/deg), `control_gripper`. Has a CLI agent for Claude, Gemini or GPT. | No numbers | Low–med. Claude servos the arm by eye in small steps, which is slow and costly in image tokens. The README says a later LeRobot update broke the original code. | 86★, last push 2025-08, V |
| [MostlyKIGuess/so101-code-as-policy](https://github.com/MostlyKIGuess/so101-code-as-policy) | Claude Code drives a real SO-101 with one webcam. OWLv2 + SAM 2.1 + homography to table cm; MuJoCo twin; self-checking skill library. **Closest to (b) + Claude.** | One 4-step run in 2 min 39 s; author: "you still have to guide it" | Med. The page says the full setup will be open-sourced later. | 0★, 2026-10-24, V |
| [Ekumen-OS/GeminiRobotics-SO101](https://github.com/Ekumen-OS/GeminiRobotics-SO101) | Gemini Robotics-ER 2 (streaming) runs pick-and-place on SO-101. Tools: `move_to_spatial_point([y,x], z)`, open/close gripper. Overhead camera, ArUco mat, ArUco tag on the jaw. Has a MuJoCo sim mode. | No numbers | Low–med. Needs a printed mat and a Gemini key. | 1★, 2026-10-31, V |
| [google-gemini/robotics-pointing-sample](https://github.com/google-gemini/robotics-pointing-sample) | Official SO-101 workshop: Gemini point → ChArUco homography → MuJoCo IK. **Points only, no grasp.** Default model IDs (ER 1.5) are shut down. | No numbers | Low. A good scaffold for pixel → table → IK. | 18★, V |
| [PAN-Chuwen/robomcp](https://github.com/PAN-Chuwen/robomcp) | **Sim only.** MuJoCo SO-101 with 22 MCP tools that return failure codes. Compares 9 agent harnesses on Gemini Flash. | Delayed slip: bare 8/15, with re-observe 15/15. Clutter: hierarchical 19/20, bare 10/20. | Design evidence only | 0★, 2026-11-03, V |
| [phosphobot](https://github.com/phospho-app/phosphobot) + [phospho-mcp-server](https://github.com/phospho-app/phospho-mcp-server) | UI/API to train ACT, SmolVLA, π0.5 and GR00T on SO-100/101. The MCP server exposes a camera frame and **replay of recorded episodes**. AI chat goes through phospho's cloud. | No numbers | Low to install. Replay skills are weak, and the cloud dependency is fading. | 393★, V |
| [Jarvis, LeRobot Hackathon Berlin 2026](https://github.com/GermainGirndt/HuggingFace-LeRobot-Hackaton-Berlin-2026) | Whisper → GPT-4o function calling → `lerobot-rollout` of a SmolVLA checkpoint (skill router). 1 of 4 skills implemented. | No numbers | Pattern only | 0★, V |
| Skill routers: [AI Bartender](https://www.elchinaslanli.com/projects/how-we-won-the-lerobot-hackathon-2025-meet-the-ai-bartender), [voice→ACT](https://github.com/Thach2001-git/SO101-Voice-Control-Imitation-Learning), [RoboticsHackCal](https://github.com/MaxBingham/RoboticsHackCal) | The LLM or a spoken keyword picks one of N trained ACT policies | No numbers | Pattern only | V |
| [Seeed reSpeaker + SO-ARM](https://wiki.seeedstudio.com/respeaker_flex_soarm/) | Wake word → Whisper → Llama 3.1 → TTS. The LLM picks a **predefined pose**; no camera, no picking. | No numbers | Not a fetch system | V |
| [Tibe31/so101-mcp](https://github.com/Tibe31/so101-mcp) | Joint-level MCP tools with limit checks; no camera | No numbers | Too low-level | 2★, V |
| [saorsa](https://github.com/saorsa-labs/saorsa) | README claims voice control of an SO-101 | — | **Not usable.** The code drives Dynamixel XM430 servos, not Feetech. | V (code inspected) |
| Your HackMIT "Pam" ([README](../../hackmit/README.md)) | Deepgram Voice Agent + function calling, YOLOE-11s-seg, Claude VLM memory, ACT on a Damiao arm | — | Reuse the YOLOE prompts and weights and the tool patterns. Drop the Voice Agent API (§3). | local |

### 1b. Language-conditioned VLAs on SO-100/101

| Model | Reported result (setting) | Language following | Status |
|---|---|---|---|
| [SmolVLA](https://arxiv.org/html/2506.01844) | SO-100, 50 demos per task: 78.3% average (π0 61.7%, ACT 48.3%). SO-101 Lego pick-place: 90% in-distribution, 50% out-of-distribution. | **Not tested** | V |
| [Yu & Qiu, Hiroshima Univ.](https://arxiv.org/html/2606.08881v2) | SO-101, 100 demos per task, 20 trials each. π0.5 56.25%, Wall-X 51.25%, ACT 33.75%, SmolVLA 32.5%. "Execution instability" is the main failure. | **Selective Color Sorting: 0–10%** for all four models | V (I opened it) |
| [MolmoAct2-SO100_101](https://arxiv.org/html/2605.02881v1) ([repo](https://github.com/allenai/molmoact2)) | **Zero-shot**, 5 pick-place tasks, 15 trials each: 56.7%, vs π0 45.3% and SmolVLA 2.3% | Not reported | V. Needs a CUDA server. You have one: [policy.sh](../so101/README.md) tunnels to the H200. |
| [GR00T N1.5 SO-101 tuning](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) | Prompt-steered table cleanup; fine-tune needs ~25 GB VRAM | Claimed, not measured | V |
| Community multi-task SmolVLA cards ([1](https://huggingface.co/khanhnd61/smolvla_so101_left_arm_3tasks), [2](https://huggingface.co/twanghcmut/SmolVLA-SO101-LeftArm-Multitask)) | Instruction text must match training verbatim; no real-robot trials | Unmeasured | V |

### 1c. Reference points on other robots

| System | What it shows | Status |
|---|---|---|
| [OK-Robot](https://arxiv.org/abs/2401.12202) (Stretch) | Open-vocabulary detect → grasp → drop: 58.5% across 10 homes, 82% in decluttered scenes. Top failures: wrong object 9.3%, hard grasp 8.0%, hardware 7.5%. Transparent objects excluded as "clearly adversarial". | V |
| [Anthropic Project Fetch, phase 2](https://www.anthropic.com/news/project-fetch-phase-two) (quadruped) | Claude was much faster than humans at bring-up and code, but "control was poor" on the actual fetch. **Keep closed-loop motion in code, not in the LLM.** | V |
| [piper-llm-arm](https://github.com/izhangfly/piper-llm-arm) (AgileX PiPER) | VLM point → 4-point table calibration → bounded skills. A governor checks preconditions, and grasp success is read from the gripper. Good design reference. | V |
| [Gemini Robotics-ER 2](https://aistudio.google.com/docs/robotics-overview) | `gemini-robotics-er-2-preview`: points as `[y,x]` normalized 0–1000, plus boxes and trajectories. ER 1.5 and 1.6 are shut down. **Free tier is "Free of charge"**; paid is $1 in / $5 out per MTok until 2026-12-31, then $2 / $10 ([pricing](https://aistudio.google.com/docs/pricing)). | V (I opened both pages) |
| [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server), [dimOS](https://github.com/dimensionalOS/dimos) | General LLM↔robot bridges. SO-101 would need ROS 2 + MoveIt, or isn't supported. High effort on a Mac. | V |

**Takeaways**

1. No hardware success rates exist for multi-step LLM fetching on SO-101. The only controlled agent numbers come from simulation (robomcp). There, re-observing before declaring "done" and returning failure codes from tools mattered more than the model.
2. Every recent SO-101 build where an LLM plans and the robot executes uses the same pattern: ground the object (VLM or detector) → table coordinates → IK primitives. That pattern is (b).
3. Do not rely on a fine-tuned VLA to follow different instructions. A skill router (one policy per skill, chosen by Claude) avoids the problem, but each skill still needs 50–100 demos.

---

## 2. Architecture (a) vs (b) on this arm

| | (a) Claude + VLA per step (Dream-1, SmolVLA) | (b) Claude + detection + calibrated camera + IK + scripted grasp |
|---|---|---|
| New data needed | 50–100 demos per skill ([SmolVLA](https://arxiv.org/html/2506.01844), [Hiroshima](https://arxiv.org/html/2606.08881v2)). Dream-1 zero-shot needs none but has not worked here. | None. Calibration already exists. |
| Evidence on SO-101 | 32–56% per task; 0–10% on language-conditioned grounding. Locally: Dream-1 zero-shot, best run held an object ~5 s ([log](../so101/DREAM_DEBUG_LOG.md)); SmolVLA 0/1. | No published SO-101 numbers. OK-Robot reached 58–82% with this structure on Stretch. |
| Picks the object Claude named | Weak (see language row in §1b) | Yes, as long as the detector finds it. Detection is a separate step you can test on its own. |
| Visible failures | Often silent: wrong object, repetition loops | Each step returns an error code: `not_found`, `not_reachable`, `grasp_failed`, `camera_moved` |
| Per-pick cost and latency | Hosted per-call billing (Dream credits; ~0.5 s per reply in your log), or a GPU server over the SSH tunnel | Local detection in tens of ms (U); IK in µs–ms. No per-call fees. |
| Generalizes to new objects | Needs new demos | Detector prompt change. Grasp shape limits apply (below). |
| Time to first "grab the bottle and hand it to me" | Days to weeks: collect, train, debug. Your 10k-step SmolVLA job still misses. | ~1–3 days of code (U) |

**Verdict: (b) is faster to a working fetch.** (a) stays useful as a later `pick` backend for objects the scripted grasp can't handle, measured with the same 20-trial protocol.

### Failure modes of (b), sized for your rig

Camera position from [camera_calib.json](../so101-bounce/real/camera_calib.json): 0.78 m above the base and 0.11 m behind it, f = 522 px at 640×480.

| Failure | Size here | Mitigation |
|---|---|---|
| Back-projecting a point that is above the table | Error = d·h/(H−h). For an object 0.25 m ahead (d ≈ 0.36 m from the camera nadir, H = 0.78): an **8 cm pill-bottle top errs 4.1 cm**, a **20 cm water-bottle top errs 12 cm** (D) | Back-project the mask's table-contact edge (the side nearest the camera nadir), or intersect with the plane at the object's known height. Then re-center with the wrist camera before closing. |
| Calibration itself | 1 px ≈ 1.6 mm at ~0.85 m range; 1.29 px ≈ 2 mm (D) | Not the bottleneck. **The calibration is at 640×480**: rescale K, or capture at that size (Dream runs use 1280×720 crops). |
| Camera moved | The mount moved at least 4 times since Nov 2, ~20 px between Nov 2 and Nov 4 (commits 6adea71, 281ea73) | Reuse the bounce runner's reference-frame check: refuse when the view moved more than 4 px |
| STS3215 backlash and sag | ~1° free play, ~0.9° deadband, 1.8–2.6° under load ([SO-ARM100 #134](https://github.com/TheRobotStudio/SO-ARM100/issues/134), V) → ~5 mm at 0.3 m (D) | Approach from a fixed hover pose, descend slowly, keep your sag trim |
| 5-DOF reach | The gripper's approach axis must lie in the vertical plane through the base axis. Pitch and roll are free; yaw is set by pan. | **Top-down** (any finger yaw) for low objects. **Radial side grasp** (horizontal, along the ray from the base) for upright bottles and cans. Map where each is feasible with IK offline. |
| Transparent objects | Your test set includes a clear bottle; OK-Robot excluded transparent objects | Test detection on your clear bottle in step 3 of the build order before trusting it |
| Arm hides the object | The Google sample says to clear the view before capturing | Capture only from a "camera-clear" pose |
| Gripper presses the table | In your Dream runs this cut power to all six servos ([log](../so101/DREAM_DEBUG_LOG.md), finding 4) | Floor margin checked in code, slow final descent |
| Grasp missed | — | The gripper position shows it: closed fully means empty. Your Dream log saw 0.21 instead of 0 while holding. Return `grasp_failed`; Claude re-observes. |

**Failure modes of (a):** language ignored ([Hiroshima](https://arxiv.org/html/2606.08881v2); [LIBERO-Plus](https://arxiv.org/html/2510.13626v3), sim, U); breaks when the camera moves (yours moves often); execution instability and repetition loops; no explicit success signal; per-call cost, latency and server availability.

---

## 3. Voice front end (minimal paid use)

**Activation**

| Option | Cost | False triggers | Notes | Status |
|---|---|---|---|---|
| **Push-to-talk** (hold a key; [Silero VAD v6](https://github.com/snakers4/silero-vad) trims silence) | $0 | None | Deepgram is guaranteed idle unless the key is held | V |
| macOS Voice Control phrase → key | $0 | Low | You already use this ([VOICE.md](../so101/VOICE.md)). A phrase such as "robot listen" could toggle recording. Hold-to-talk is not possible this way. | U (untested) |
| [openWakeWord](https://github.com/dscripka/openWakeWord) | $0 | Some | Apache code, but pre-trained models are CC BY-NC-SA. Custom word trains in under 1 h (Colab). On macOS use `inference_framework="onnx"`. | V |
| Porcupine | — | — | [FAQ](https://picovoice.ai/docs/faq/general/): no personal or non-commercial plans. Free AccessKeys disabled 2026-06-30 per a [user-quoted email](https://community.home-assistant.io/t/fyi-picovoice-confirmed-free-tier-accesskeys-will-stop-working-after-june-30-2026/1012744). | V (FAQ), secondhand (email) |

**Speech to text**

| Engine | English WER | Latency for a short command | Notes | Status |
|---|---|---|---|---|
| **[parakeet-mlx](https://github.com/senstella/parakeet-mlx)** (parakeet-tdt-0.6b-v2) | 6.05% Open ASR average ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)) | 0.50 s on an M4 (one test, [speedtest](https://github.com/anvanvan/mac-whisper-speedtest)) | Apache code, CC-BY weights, 993★, active | V |
| mlx-whisper (large-v3-turbo) | — | 1.02 s (same test) | MIT | V |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Core ML) | — | 1.23 s (same test) | MIT; built-in Silero VAD | V |
| Apple SpeechAnalyzer via [`yap`](https://github.com/finnvoor/yap) | 2.12% LibriSpeech clean (vendor) | — | On-device; a CLI streams text to stdout | U (WER) |
| Deepgram [Nova-3 / Flux](https://deepgram.com/pricing) streaming | — | ~260 ms end-of-turn (Flux, vendor) | **$0.0077/min**, billed per second of processed audio. $200 free credit. | V |

- **Deepgram cost:** 30 commands a day at 4 s each is ~$0.46/month (D). Cost is not the issue; the gating is. Open the socket on key-down and close it on key-up.
- **Avoid the Deepgram Voice Agent API** (what Pam used). It is $0.075/min, "calculated on WebSocket connection time", so idle sessions cost money. You don't need its LLM anyway, since Claude is the LLM.
- **TTS:** keep `say`. [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache) is the upgrade if the voice matters. Piper is now GPL. Deepgram Aura-2 is $0.030 per 1k characters.

**Recommendation:** push-to-talk + parakeet-mlx, so all speech recognition is local and free. Deepgram Flux stays an opt-in switch, gated by the same key.

---

## 4. Claude as the planner

### Which surface

| Surface | Fit | Notes |
|---|---|---|
| **Agent SDK, `ClaudeSDKClient` (Python)** | Best | In-process tools via [`@tool` + `create_sdk_mcp_server`](https://code.claude.com/docs/en/agent-sdk/custom-tools). One session across commands. `interrupt()`. Hooks, `can_use_tool`, `max_budget_usd`. Cost reported in `ResultMessage.total_cost_usd` ([ref](https://code.claude.com/docs/en/agent-sdk/python)). V |
| `claude -p` + `--mcp-config` | OK | The robot server becomes a separate stdio process. `--bare` skips your `~/.claude` hooks and CLAUDE.md but needs `ANTHROPIC_API_KEY` ([headless](https://code.claude.com/docs/en/headless)). Pays a cold start on every command unless kept alive with `--input-format stream-json`. V |
| Anthropic API tool runner | Leanest | Not the Claude Code harness you asked for; fallback only |

**Auth:** the SDK docs document API-key auth, and third-party products may not offer claude.ai login ([overview](https://code.claude.com/docs/en/agent-sdk/overview), V). Whether your personal subscription login works through the SDK is **U**.

### Minimal configuration

Option names checked against the docs (V):

```python
options = ClaudeAgentOptions(
    system_prompt=PLANNER_PROMPT,           # own prompt; the default is a minimal tool-calling prompt
    tools=[],                               # remove all built-ins (Bash, Edit, Web...)
    mcp_servers={"robot": robot_server},    # create_sdk_mcp_server(name="robot", tools=[...])
    allowed_tools=["mcp__robot__*"],
    permission_mode="dontAsk",              # anything not allowed is denied, never prompted
    setting_sources=[],                     # do not load ~/.claude CLAUDE.md, hooks, settings
    max_turns=10, max_budget_usd=0.50,
    env={"ENABLE_TOOL_SEARCH": "false"},    # ~6 tools: load upfront, skip the search round-trip
    include_partial_messages=True,          # stream text so `say` can start early
    hooks={"PreToolUse": [HookMatcher(matcher="^mcp__robot__", hooks=[log_call], on_failure="block")]},
)
```

- **`setting_sources=[]` matters here.** By default the SDK loads user, project and local settings. That would bring in your global CLAUDE.md and your `no-write-ops` / `futurama-guard` hooks on every call ([ref](https://code.claude.com/docs/en/agent-sdk/python)).
- **Tool search is on by default** and adds a round-trip. The docs say loading upfront is typically faster below ~10 tools ([tool search](https://code.claude.com/docs/en/agent-sdk/tool-search)).

### Tool surface

Keep tools coarse, so Claude makes 3–5 calls per command, not dozens of jog moves as with robot_MCP.

| Tool | Returns | Moves the arm |
|---|---|---|
| `look()` | Annotated 640×480 frame plus a detection list: id, label, table x/y, size, whether reachable | No |
| `find(query)` | Best match, confidence, reachable or not | No |
| `pick(object_id)` | `ok`, or `not_reachable` / `grasp_failed` / `user_declined` / `camera_moved` | Yes |
| `place(target)` | `ok` / error | Yes |
| `hand_over()` | Holds at a fixed handover pose; opens on "got it" or a key | Yes |
| `home()` | `ok` | Yes |

**"Make a sandwich":** the planner prompt should make Claude say which steps no tool covers (spreading, cutting) instead of improvising. A 0.4 m two-finger arm can fetch and place the ingredients; it cannot assemble a sandwich.

### Cost per command (D)

Assumptions:

- One command = 4 model turns.
- ~5k tokens of system prompt and tools (cached after turn 1), ~1k new tokens per turn, and one 640×480 image (~410 tokens).
- ~800 output tokens per turn, including thinking.
- Prices from [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing) (V).

| Model | $/MTok in / out / cache hit | ≈ $ per command |
|---|---|---|
| Opus 5.5 | 4 / 20 / 0.20 | ~0.10 (0.05–0.25 depending on thinking) |
| Sonnet 5.5 | 2 / 10 / 0.10 | ~0.05 |
| Haiku 5.5 | 0.10 / 0.50 / 0.01 | ~0.003 |

Measure the real figure from `total_cost_usd` and `model_usage` on 10 scripted commands before choosing.

### Latency per command (all U, measure with `duration_api_ms`)

| Piece | Expected |
|---|---|
| Key release → transcript (parakeet-mlx) | 0.2–0.5 s |
| SDK session start | Once per session with `ClaudeSDKClient` |
| Model turns | ~1–5 s each × 3–5 turns, so 5–20 s of planning. Lower `effort`, coarse tools and a cheaper model all cut it. |
| Detection (YOLOE-26s on a Mac) | Tens of ms; no published MPS figures |
| Motion: pick + handover at your 35°/s cap | 10–20 s |

Speak an acknowledgement with `say` as soon as the transcript arrives, so the planning pause doesn't feel dead.

### Safety layers

| Layer | Mechanism |
|---|---|
| What Claude can call | `tools=[]`, `permission_mode="dontAsk"`, only `mcp__robot__*` allowed, `setting_sources=[]` |
| **Confirmation before motion** | **Inside the motion tool, after perception.** The tool shows the annotated frame, says "Picking the clear bottle, 24 cm ahead, 6 cm left. Say go." and waits for "go" or a key. A `can_use_tool` callback would only see Claude's argument ("bottle"), not the resolved coordinates. |
| Workspace limits (in code, not the prompt) | Reach annulus; floor margin above the table; keep-out zone on your side except the handover pose; 10° from joint ends; 35°/s cap; 50% torque limit (the last three already exist in your Dream setup) |
| Budget | `max_turns`, `max_budget_usd` per command |
| Stop | **Hardware:** a switch on the servo supply (the arm drops). **Software:** a key watched by the motion thread, not the LLM, that holds and glides home ([home.py](../so101/home.py) pattern). **Agent:** `client.interrupt()`. Never rely on voice for the e-stop. |
| Camera moved | Refuse picks when the view moved more than 4 px from the reference frame |
| Audit | A `PreToolUse` hook logs every call, with `on_failure="block"` so a crashed guard blocks the call |

---

## 5. Recommended stack and build order

| Layer | Choice | Cost |
|---|---|---|
| Activation | Push-to-talk (Silero VAD trim); openWakeWord later if you want hands-free | $0 |
| STT | parakeet-mlx (tdt-0.6b-v2); Deepgram Flux opt-in, key-gated | $0 (Deepgram ≈ $0.0077/min) |
| TTS | `say` | $0 |
| Planner | Claude Agent SDK, `ClaudeSDKClient`, configured as in §4 | ~$0.003–0.10 per command (D) |
| Detection | YOLOE-26s text prompts on the Mac (AGPL; fine for personal use). Gemini ER 2 free tier for referring phrases ("the mug left of the laptop"). | $0 |
| Geometry | [camera_calib.json](../so101-bounce/real/camera_calib.json) back-projection to the table plane | $0 |
| IK | LeRobot [`RobotKinematics`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py) (placo, `lerobot[kinematics]`, not installed yet) or [so101ik](https://github.com/stepankandrushin/lerobot-so101-inverse-kinematics) (1★, new). Check both against your FK ([bounce/robot.py](../so101-bounce/bounce/robot.py)). | $0 |
| Grasp | Scripted: hover → descend → close → check gripper position. Top-down for low objects, radial side grasp for upright ones. | $0 |
| Optional | Learned `pick` backend: MolmoAct2 or SmolVLA over the H200 tunnel, or Dream-1 | GPU time / credits |

| # | Build | Pass criterion | Cheapest test (no robot, no paid API) |
|---|---|---|---|
| 1 | Robot tools with a dry-run backend that logs targets and draws them on the frame | Handlers callable directly; error codes returned | pytest calling the async handlers; no Claude |
| 2 | Pixel → table back-projection | The 16 calibration points reproject within ~1.3 px; tape-mark targets within 1 cm | Unit test against the stored calibration points (CPU, seconds) |
| 3 | Detection | Right object in at least 19 of 20 saved overhead frames; contact point within 1 cm of a hand label; **include the clear bottle** | YOLOE-26s locally on saved frames; Gemini ER 2 free tier as a comparison |
| 4 | IK + reachability map | FK(IK(p)) within 2 mm over a grid; map of top-down and side-grasp feasibility | placo or so101ik on `so101_new_calib.urdf`, CPU |
| 5 | Sim check | Grasp and handover paths stay above the table and inside joint limits | MuJoCo [so101.xml](../so101-bounce/assets/robotstudio_so101/so101.xml) |
| 6 | Claude planner on dry-run tools | Correct tool sequence for 10 scripted commands, including an infeasible one; cost per command logged | **The only paid step.** About $0.03 for 10 commands on Haiku 5.5, about $1 on Opus 5.5 (D). |
| 7 | Voice | Correct transcripts for 20 recorded commands; Deepgram never opened without the key | parakeet-mlx on WAV files; `say` |
| 8 | Real arm, hover only (**ask the user first**) | 10 hovers 5 cm above detected objects, error under 1 cm | — |
| 9 | Real grasp + handover (**ask first**) | 20 trials, bottle at random reachable spots: success rate plus failure taxonomy | — |
| 10 | Optional learned `pick` backend | Same 20-trial protocol, compared with step 9 | Serve on the H200 through the existing [policy.sh](../so101/README.md) tunnel |

## Open questions

- Gripper maximum opening, and the radius where top-down grasps are feasible: measure, or compute in step 4.
- Detection quality on the clear bottle: step 3 answers it.
- Whether the Agent SDK can use your Claude subscription login instead of an API key.
- Real Claude latency per turn at each effort level: step 6 answers it.
- Dream-1 fine-tune status and per-call economics: covered by the agent reading Dream's docs, not this survey.
