# Dream-1 docs brief: voice "fetch it" agent on one SO-101

All 35 Dream Labs customer docs, read 2026-11-05 through the dreamlabs-docs MCP server. The raw Markdown is in
[scratchpad/dream/docs/](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/dream/docs/)
(session scratch, may be wiped). Citations are `doc-id §section`. Lines marked **(SDK)** come from the library source:
the installed 1.0.10 / 1.0.5, diffed against the PyPI wheels dreamlabs 1.0.23 / dreamlabs-client 1.0.14. The reference
docs were generated from 1.0.23 (reference/index §dream.\_\_version\_\_). No Dream calls were made.

## Summary

| # | Fact | Source |
|---|---|---|
| 1 | The task is free text in `observation["prompt"]`. There is no goal image, no "done" flag and no success detection. | control-loop §What you send; examples/yam-mujoco README (shipped) |
| 2 | On any arm other than a YAM, the server may ignore the joints, hand pose and grip, so the model acts only on the cameras and the prompt. | control-loop §What the model reads on another arm |
| 3 | One arm means: run it as the right arm, park the left with grip 1, send a black `video.wrist_l`, never play `q_l_chunk`, and name only the right arm in the prompt. | control-loop §On another arm (bullet "One arm") |
| 4 | `dreamcli` still does not take a single SO-101 on macOS. 1.0.23 requires both `follower_l` and `follower_r`, and a `/dev/serial/by-id` or `by-path` port. | index §2; **(SDK)** robots/config.py |
| 5 | One `get_action()` returns one chunk of 48 rows. The model learned rows 1/30 s apart. By default a new chunk is asked for every 30 rows. | control-loop §Playing the reply, §Running it in a loop |
| 6 | Billing is per successful `get_action` and failed calls are free. The docs give no price. An API key cannot read the balance. | index §1; fine-tuning §What it costs |
| 7 | Rate limit: 3 calls in quick succession, then 1 every 0.7 s, per session. | control-loop §Running it in a loop; fine-tuning §Handle startup |
| 8 | Call `reset_episode()` before every new task. Changing the prompt does not clear the model's memory of the episode. | control-loop §Episodes |
| 9 | Dry runs call the model for real (billed), and so do tasks in the YAM sim. Only the sim's offline commands make no calls. | robot-of-your-own §Run your robot; examples/yam-mujoco §Install and start |
| 10 | Fine-tuning costs $200 and takes 7–8 h with no cancel. A one-arm robot can only fine-tune in joint space ("Single-arm end-effector bodies are not supported"). | fine-tuning §What it costs, §End-effector pose |

## 1. How Dream-1 takes a task

### Inputs: `set_observation(frames, observation, *, timestamp=None)`

| Key | Dict | Shape | Value | Source |
|---|---|---|---|---|
| `prompt` | `observation` | str | "The task in words, for example `put the cup on the plate`." | control-loop §What you send |
| `state.<side>_arm` | `observation` | joints | Joint angles in radians, base to flange | control-loop §On another arm |
| `state.<side>_grip` | `observation` | 1 | 0 closed, 1 open. On another arm, send it "in place of `state.<side>_gripper`" (route A) | control-loop §On another arm |
| `state.<side>_eef_xyz`, `_eef_rot6` | added by the client | 3, 6 | With `ik`, the client computes them by FK. Sending them yourself is refused. | control-loop §What you send (note) |
| `embodiment` | added by the client | str | `other_embodiment` on a non-YAM arm. "Never send it: an observation that carries it is refused." | control-loop §What you send |
| `video.overhead`, `video.wrist_l`, `video.wrist_r` | `frames` | 1 base64 JPEG each | The client keeps the last 9 per camera, 0.2 s apart (5 Hz), and sends them oldest first | control-loop §What you send, §Which frames are sent |

There is no goal-image input. Every key is a state, a camera or `prompt`, and "any other key is refused as
`malformed`" (control-loop §What you send).

### Reply: `get_action(*, commanded=None) -> dict`

| Key | Shape | Value |
|---|---|---|
| `q_r_chunk`, `q_l_chunk` | 48 × (joints + 1) | Absolute joint targets in rad, then the gripper (0 closed–1 open). With `ik` head `eef` they are IK-solved from the model's hand poses. They are already smoothed and blended onto the previous chunk: "Do not blend the seams again." |
| `per_key["action.<side>_eef_xyz"]` | 48 × 3 | TCP position, m, absolute, in the arm's base frame |
| `per_key["action.<side>_eef_rot6"]` | 48 × 6 | First two columns of the rotation matrix `[r00, r10, r20, r01, r11, r21]` |
| `per_key["action.<side>_grip"]` | (48,) | 0 closed–1 open |
| `per_key["action.<side>_arm"]` | null | On a non-YAM arm, a base model answers no joints |

Source: control-loop §What you get back, §Playing the reply, §What the model reads on another arm. Row 0 is the target
at the time of the newest observation sent; each row after is one step later. "Every chunk is 48 rows … It is not a
setting." The docs name no other reply key ("Read the two chunks; ignore any other key"). In particular there is no
done or progress field.

### Timing and calls

| Item | Value | Source |
|---|---|---|
| Learned row spacing | 1/30 s, so 48 rows = 1.6 s | control-loop §Playing the reply |
| `dreamcli` playback | 20 rows/s, so 2.4 s per chunk | control-loop §Playing the reply |
| Refill | `refill_due(executed, remaining)`. Async while calls take ≤ 0.7 s: the reply lands after 30 rows (`execution_horizon`). Sync after a slower call: the arm holds until the chunk runs out. | control-loop §Running it in a loop |
| Pacing keys | `execution_horizon` (30), `action_hz` (20), `sync_above_ms` (700) | reference/dream-client §policy |
| Round trip | 250–450 ms once the model is running | fine-tuning §Handle startup |
| Calls per pick-and-place | **Not stated in the docs.** At the defaults it is 1 call per 1.5 s (40 per minute). Our runs at 30 rows/s made about 1 call per second (83 calls in 83 s, run 20261021-223439, [DREAM_DEBUG_LOG](../so101/DREAM_DEBUG_LOG.md)). | derived |

One call returns one chunk. `get_action()` runs one call at a time, and a second concurrent call raises `DreamError`
(control-loop §The calls).

### Prompt phrasing the docs use or recommend

- Examples: `put the cup on the plate`, `put the cup back`, `pick up the headphones`, `put apple into the bowl`,
  `put orange into the bowl`, `put banana on the plate` (control-loop, fine-tuning §Run the loop, examples/yam-mujoco
  §Give the arms a task). All are short imperative pick-and-place commands with one object and one target.
- One arm: "name only the right arm in the prompt" (control-loop §On another arm).
- Fine-tunes: "The model learns to follow the task text, so write each task the way you'll prompt the robot"
  (fine-tuning §Check the dataset).
- New task: `reset_episode()` first, since "changing the prompt alone does not clear the model's memory of the
  episode" (control-loop §Episodes).

## 2. Single arm and SO-101

### What "single-arm support" is in the docs

Verbatim (control-loop §On another arm):

> **One arm.** Run it as the right arm. Send the left as a fixed parked pose with grip 1 and a black `video.wrist_l`,
> never play `q_l_chunk`, and name only the right arm in the prompt.

> **Cameras.** Exactly three: one scene camera (`video.overhead`) and two wrist cameras (`video.wrist_l`,
> `video.wrist_r`). For a camera your robot does not have, send a black frame, never another camera's view.

[so101/dream_run.py](../so101/dream_run.py) already does this. The docs describe no dedicated single-arm mode, and
nothing in 1.0.23 adds one **(SDK)**.

| Question | Answer | Source |
|---|---|---|
| Does `dreamcli` run one SO-101 on macOS? | **No.** The docs say "It needs Linux (Ubuntu) and Python 3.12 … an SO-101 needs 3.12". In 1.0.23 the station config still requires both `follower_l` and `follower_r` (`_require(raw, key, "")`), the SO-101's `can_channel` must start with `/dev/serial/by-id/` or `/dev/serial/by-path/`, and the beta robot is listed as "XLeRobot (two SO-101)". | index §2; **(SDK)** robots/config.py `_read_arm`, `_read_so101_arm`; robots/beta_robots.json |
| SO-101 driver | `so101.py`: 5 joints + gripper on STS3215 over LeRobot's Feetech bus. Joints in rad in the URDF frame, "dream's `so101_faas` model, one URDF per side", 0 at the LeRobot calibration's middle pose. The gripper is linear from `range_min` (0) to `range_max` (1). It runs at 60 Hz and, before torque-off, parks where the arm was opened at 20°/s. | drivers §The SO-101 |
| IK options (route A) | `"ik": {"arm": ..., "head": "eef"}`, with `arm` = `"metal"`, `"makers"`, or an arm file (`urdf`, `base`, `flange`, `tcp`, `approach`, `jaw`, `rest`). `so101_faas` is Dream's built-in SO-101 spec (named in drivers §The SO-101, not in the policy docs). Head `joint` with `other_embodiment` is refused. | reference/dream-client §policy; control-loop §Two ways to run your arm |
| IK option (route B) | No `ik`. Set `"robot": "other_embodiment"`, send six zeros as `state.<side>_arm`, send your own FK pose, then solve `per_key` poses yourself. | control-loop §Two ways to run your arm |
| Tool center point | "Agents: stop here." The user measures it with `dreamcli tcp`. An agent never picks or estimates it. "a TCP 2 cm off puts every grasp 2 cm off, and nothing reports it." `policy()` warns if the arm file's TCP was not set with `dreamcli tcp`. | control-loop §On another arm; reference/dream-client §policy |
| Joint state on SO-101 | Sent, but "the serve may therefore ignore the request's state … and the model acts on your cameras and the prompt". The hand pose still anchors the answer. The 1.0.5 docstring was blunter: "it reads all zeros" **(SDK)**. | control-loop §What the model reads on another arm |
| Camera names | The base model reads `overhead`, `wrist_l`, `wrist_r` (`Model.cameras`). With `ik`, the client renames the body's single scene camera to the model's scene role, so the `video.scene` key in dream_run.py is sent as `overhead` **(SDK)** dream/client_ik.py `camera_slots`. | reference/model |
| Frame format | 320×176 base64 JPEG at quality 85, no `data:` prefix, encoded from BGR. Center-crop to 16:9 first. Any other size is stretched, then 2.5% is trimmed from each edge. Use the same cameras all episode. | control-loop §Each frame |
| SDK version | The service accepts `dreamlabs-client` ≥ 1.0.0; 0.x is refused with `client_outdated`. The installed 1.0.5 qualifies. Python 3.12 is needed for an SO-101, and `ik` needs Python 3.11 or newer. | index §2; control-loop §On another arm |

Conflict to note: the docs say to run the arm as the right arm, but our best zero-shot run was as the left arm
(DREAM_DEBUG_LOG finding 5). dream_run.py now defaults to `--side left`.

### Changes from 1.0.10 / 1.0.5 to 1.0.23 / 1.0.14 (SDK diff; the docs have no changelog)

| Area | Change |
|---|---|
| `dream/client.py`, `client_ik.py`, `facade.py` | Docstring wording only: state "may be ignored" instead of "withheld / all zeros" |
| `robots/station.py`, `setup_*.py` | Compile the `traj` IK solve before the arms are powered (a Metal CAN-timeout fix). A station file without a `checkpoint` now warms up with the first listed model. |
| `robots/sys0/*`, `yam.py` | YAM System-0 tracker (YAM only) |
| `robots/so101.py`, `feetech.py`, `adapter.py`, `beta_robots.json`, `config.py` logic | Unchanged |

Upgrading is not needed for this project, because nothing on the SO-101 path changed. The exception is a call refused
with `client_outdated`, which can happen "after a deploy" (reference/dream-bad-request).

## 3. Costs

| Item | Answer | Source |
|---|---|---|
| Price per `get_action` | "every successful `get_action` call costs the price the Billing page states for the model it ran (a fine-tuned model costs what its base model costs) … failed calls cost nothing". **No number in the docs.** | index §1 |
| Per-second pricing | None. Billing is per call, so cost is roughly (seconds run ÷ seconds per chunk) × price. | index §1 |
| Free tier | "Every account starts with free credit". The amount is not stated. "Keys stop when the credits run out". Auto-reload is on the Billing page. | index §1 |
| Check credits from code | **Not possible**: "Your balance is on the Billing page; an API key can't read it." The only in-band signals are `DreamAuthError` reason `over_cap` (the monthly budget cap) and keys stopping. | fine-tuning §What it costs; reference/errors §DA-1000 |
| Budget cap | Set a monthly cap on Billing; over it, `DA-1000 over_cap` pauses the keys. | reference/errors §DA-1000 |
| Rate limits | Per session, 3 calls in quick succession then 1 per 0.7 s, otherwise `DA-2000 rate_limited` (`retry_after_s` 0.7). One `get_action` at a time per client. | control-loop §Running it in a loop; fine-tuning §Handle startup |
| Cold start | `pool_warming` (a free refusal). `dream-1-beta` is "normally kept running". A cold model can take up to about 30 min; a fine-tune takes up to 1 h ("25 to 50 minutes" in their tests), including after it has been idle. | control-loop §Try one call first; fine-tuning §Handle startup |
| Billed without moving | `dreamcli robot run --dry-run`: "The model is called for real and nothing is written to the arms". So is our dream_run.py dry run, and its `wait_until_serving()` makes one successful (billed) call before every run. | robot-of-your-own §Run your robot; [dream_run.py](../so101/dream_run.py) |
| Fine-tune | "$200 at the time of writing", held from credits at queue time and charged when published. A run that stops short is prorated. No cancel. While the hold is open, keys can spend only the credit above it. Training takes about 7–8 h plus queue time, on at most 3 cameras. | fine-tuning §What it costs, §How long it takes |

## 4. Dream YAM Sim (`dreamcli example yam-mujoco`)

| Question | Answer | Source |
|---|---|---|
| What it is | A local MuJoCo scene: two YAM arms, three RGB cameras, fruit, a bowl and a plate. You type a task and Dream returns joint chunks. It uses the YAM, not the SO-101. | examples/yam-mujoco |
| Billed? | Yes: "Task submissions use account credits." Each request is a real `get_action`. | examples/yam-mujoco §Install and start |
| Free modes | `dreamcli example yam-mujoco --self-test` (no display, no key). From the shipped README: `python yam_env.py --headless --frames frames` and `python dream_control.py --dry-run --headless` "make no inference calls". The README also lists `mock_pickups.py` and unit tests, which the wheel does not ship. | examples/yam-mujoco §Install and start; README in `dream_examples/yam_mujoco` **(SDK)** |
| Language-conditioned tasks? | Yes, on the YAM: `put apple into the bowl`, `put orange into the bowl`, `put banana on the plate`; Tab lists recommendations. `/model dream-1-ft:...` runs a fine-tune. | examples/yam-mujoco §Give the arms a task |
| Defaults | `dreamlabs/dream-1-beta`, joint control, 48 actions per response (`/actions 24` changes it), 120 s trial (`/seconds`). The sim holds while inference is pending. It plays 20 rows/s **(SDK)** `dream_control.py` `"action_hz": 20`. | examples/yam-mujoco §Give the arms a task |
| Install / run | `source ~/dream-env/bin/activate && dreamcli example yam-mujoco`. It needs a display, and on macOS it launches `mjpython` itself (README). | examples/yam-mujoco §Install and start |
| Limits | Grip friction and object masses are tuned, not calibrated. "Banana-to-plate remains inconsistent." "passing a scripted pickup does not establish model task reliability". The README adds: "Task success is not automatically detected" and "Live apple/orange-to-bowl trials succeeded once each". | examples/yam-mujoco §Simulation limits; README |

## 5. Live View, episodes, safety, errors

### Live View (`dream.LiveViewClient`)

| Call | What it does | Source |
|---|---|---|
| `LiveViewClient(robot_id, *, dream_client=client, face_blur=True, model_fixed=True, cameras=None)` | `robot_id` is `rbt_` + 16 hex. Pass `dream_client`, otherwise the page shows "Dream-1 is getting ready" all run. | reference/live-view-client §Constructor |
| `stream(obs, status, *, arms=, error=, prompt=, run=, model=, steps=)` | Sends the newest frames (640 px wide for the page, not the model's 320×176). `status` is `idle`, `running` or `error`. The call returns at once on its own thread. | live-view §The three calls |
| `get_prompt()` | The task a viewer last typed, or None | live-view §The three calls |
| `register_callback(fn, *, home=False, reset=False, prompt=False)` | `EventChange.start` / `.end` (Stop, or a new task while running) / `.prompt` / `.home` / `.reset`. Runs on the client thread, so it only records the event. | reference/event-change; live-view §Everything the station does |
| Register a robot | `curl -s -X POST "$DREAM_API_URL/v1/robots" -H "Authorization: Bearer $DREAM_API_KEY" -H "Content-Type: application/json" -d '{"name": "lab-robot", "embodiment": "yam"}'` returns `{"id": "rbt_...", ...}`. The docs do not say which `embodiment` value a zero-shot SO-101 should use. | live-view §The robot id and your key |
| Faults | Logged to `~/.dream/logs/dream.log`: `4001` unknown key, `4004` unknown or foreign robot id, `4005` revoked key | live-view §Troubleshooting |

The docs never say Live View is billed (only `get_action` and fine-tunes are). It moves no arms: Home and Reset are
answered "not supported" unless the callback takes them.

### Episodes

- `reset_episode()` before every new task or trial. It ends the server session and empties the frame windows. It is
  refused while a `get_action()` runs (control-loop §Episodes).
- "A session that is left idle ends after a short while" (the length is not stated). The next call then raises
  `DreamSessionExpired`: drop the rows, ask again, and ramp in (control-loop §Episodes; reference/errors §DA-0001).
- Calling `policy(...)` again also starts a fresh session.

### Safety rules

| Rule | Source |
|---|---|
| "Nothing moves without me. Never run Arm, a run without --dry-run, the setup wizard, or anything else that switches a motor on or sends a target." | robot-of-your-own §Write your driver (agent rules) |
| Agents never pick or estimate the TCP (`tcp`, `approach`, `jaw`); the user confirms it | index §3; control-loop §On another arm |
| `close()` and every disconnect path must never move the arm. Read a vendor's `connect()`/`disconnect()` before calling it. | drivers §The driver template |
| Ramp into every new chunk over ≥ 2 s at ≤ 0.5 rad/s per joint (at start and after a new session) | control-loop §Running it in a loop |
| End effector < 0.6 m/s; gripper ≤ 0.05 per row; keep joint limits | control-loop §Safety |
| Hold when a camera's newest frame is older than 0.2 s, or no row is due | control-loop §Safety |
| "Nothing is limited for you": no clamps on joints, speed or workspace. The IK trusts the URDF's limits. | control-loop §Playing the reply, §On another arm |
| Keep a hand on the e-stop whenever the arms are powered | control-loop §Safety |

### Errors and retries

| Error | Retry? | Action | Source |
|---|---|---|---|
| `DreamServerError` `pool_warming` (DA-2000) | yes | Wait `retry_after_s` (30 s) and keep asking. It costs nothing. | control-loop §Errors |
| `rate_limited`, `pool_full`, `queue_timeout` (DA-2000) | yes | Wait `retry_after_s` | reference/errors §DA-2000 |
| `max_serves` (DA-2000) | no | Try later | reference/errors §DA-2000 |
| DA-5000 `error` | yes, but stop after 3–4 in a row | "a model that keeps failing the same call will not recover by itself" | control-loop §Errors; fine-tuning §Handle startup |
| `DreamTimeout` (15 s `timeout_ms`), `DreamProtocolError` | yes | Hold, wait 1 s, ask again | control-loop §Errors |
| `DreamSessionExpired` | – | Drop the rows, bring the arm to a safe pose, ask again or reset | reference/dream-session-expired |
| `DreamAuthError` (`unknown_key`, `disabled`, `over_cap`, `not_entitled`, `region_not_supported`) | no | Stop | reference/errors §DA-1000 |
| `DreamBadRequest` (`malformed`, `too_large`, `unknown_checkpoint`, `client_outdated`) | no | Fix the request | reference/errors §DA-4000 |
| `DreamObservationError`, `DreamConfigError`, `DreamActionError` | no | Fix what the message names | control-loop §Errors |
| `DreamSlowNetworkWarning` | – | A warning only, issued when the median of the last 5 round trips is ≥ 1 s. Read `client.last_timing`. | reference/dream-slow-network-warning |

## 6. What changes the design of an LLM-planner → Dream-1 executor

| Docs fact | Design consequence |
|---|---|
| No done flag, no success detection, no progress field (§1; sim README) | The planner decides when a step ends: a time or chunk budget per step, then Claude checks a fresh overhead frame. |
| No built-in task decomposition; examples are single pick-and-place sentences (§1) | Claude splits the request into one-object, one-target steps in the docs' phrasing. |
| No goal image (§1) | Steps are language only. "Give it to me" has to become a place both cameras can see, e.g. "put the bottle on the plate". |
| State ignored on non-YAM arms (§2) | Only the cameras and the prompt steer the model. Camera framing (16:9, a YAM-like overhead view) and prompt wording are the levers. |
| Session memory per episode; `reset_episode()` per task (§5) | One step = one episode: `reset_episode()`, a new prompt, run, stop. Claude's thinking between steps never runs inside an episode, which also avoids idle expiry. |
| Billing per call; `execution_horizon` is configurable (§1, §3) | Steps cost (seconds ÷ seconds per chunk) × price. Playing all 48 rows at 20 Hz (`execution_horizon: 48`) means 1 call per 2.4 s, against about 1 per second in our 30 rows/s runs. It is less reactive. |
| Balance is not readable by API (§3) | Count successful calls locally per session with a hard cap, and set a monthly budget cap on Billing as the backstop (`over_cap`). |
| Live View gives Start/Stop/prompt events and `stream(prompt=, error=)` (§5) | Optional remote monitor: show the current sub-step as `prompt`, and treat `EventChange.end` as an abort. It cannot replace the voice front end or the e-stop. |
| Fine-tune: single-arm joint space, task text = prompts, `head_camera` is "the view your dense text labels come from" (fine-tuning §Config keys) | If we fine-tune, record demos per sub-step using the exact step sentences the planner emits. |

## What this means for our agent

1. **Executor contract.** `run_step(prompt, max_s)` does: `reset_episode()`, then the docs' loop with the one-arm
   layout, then stop at `max_s`, then hold, then return. Success is judged by Claude from a new overhead image, never
   from Dream's reply.
2. **Step vocabulary.** Steps are short "put/pick up the X (in/on the Y)" sentences that name only the arm Dream
   drives. Multi-stage requests like "make a sandwich" are far outside what the docs show working (their own sim
   calls banana-to-plate inconsistent). The planner should say so and offer a reduced fetch task.
3. **Credits.** Use one `DreamClient` and one `policy()` for the whole voice session. [robot.py](robot.py) currently
   launches dream_run.py per step, which pays a billed warm-up call every step. Use a local call counter with a hard
   cap, and consider `execution_horizon: 48`. Remember that dry runs and sim tasks are billed.
4. **Testing ladder.** Mock robot (free) → sim offline commands (free) → a few billed YAM-sim tasks to see language
   following → one billed SO-101 dry run → live, with the user's OK for every arm-moving run.
5. **Cold starts.** `dream-1-beta` is normally warm. A fine-tune can need 25–50 min after idle, so warm it before the
   user starts talking (`pool_warming` refusals are free).
6. **Safety stays in our loop.** Keep the speed caps, the 0.2 s stale-frame hold, the 2 s ramp, the table floor and the
   typed confirmation for live steps. Agents never power motors or set the TCP.

## Not answered by the docs

- The price per call and the size of the free credit (Billing page only).
- How long an idle session lasts.
- Whether Live View costs anything, and which `embodiment` value to register a zero-shot SO-101 under.
- `ik.region` (the table floor dream_run.py uses) exists in the client source but is not documented.
