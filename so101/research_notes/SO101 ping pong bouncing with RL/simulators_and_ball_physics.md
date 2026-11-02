# Simulators, SO-101 assets, and ping-pong-ball physics for GPU-parallel RL paddle bouncing

Scope: picking a simulator and an SO-101 model for training a state-based policy that bounces a 40 mm / 2.7 g ball on a table-tennis paddle held in the SO-101 gripper, then deploying it on a real arm. Researched 2026-10-30. Versions are noted inline.

**How to read the "own measurement" citations.** Several findings come from small CPU experiments I ran locally (MuJoCo 3.17.0 and MJX with JAX 0.12.0.1 on CPU, macOS, one ball, each run under 10 s). The scripts are linked as sources. They sit in the session scratchpad, so copy them if you want to keep them:
[bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py),
[bounce_test2.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test2.py),
[bounce_test3.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test3.py),
[bounce_test4.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test4.py),
[bounce_test5.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test5.py).
In every test the ball is a 2.7 g sphere of radius 20 mm and the paddle is a 160 × 150 × 10 mm box. COR means rebound speed divided by impact speed, measured on the first bounce.

---

## 1. SO-101 / SO-100 simulation assets: which are usable, and how good are their mass and actuator models?

### Takeaway
Use the MuJoCo Menagerie model `robotstudio_so101` (MJCF) or the TheRobotStudio URDF it was derived from. Every SO-101 asset found shares the same CAD-derived link masses and joint limits. None has an identified actuator model for a paddle-loaded arm moving fast. The servo gains differ widely between assets: Menagerie kp ≈ 998 with a ±2.94 N·m force range; Isaac `isaac_so_arm101` stiffness 50–200 with a 1.9 N·m effort limit and a 1.5 rad/s velocity limit; ManiSkill has SO-100 only, with a 100 N·m force limit. Plan to system-identify the actuators yourself. ManiSkill3 has no SO-101 model, and LeRobot ships no SO-101 simulator.

### Cited Findings
**MuJoCo Menagerie `robotstudio_so101`** (google-deepmind/mujoco_menagerie)
- The folder exists alongside `trs_so_arm100` (SO-100). It contains `so101.xml`, `scene.xml` and `scene_box.xml`. It was added 2025-12-19. A camera-mount PCB was added 2026-09-23, and the `wrist_roll` upper limit was fixed 2026-09-28 (PR #324). — [Menagerie commits for robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/commits/main/robotstudio_so101); [CHANGELOG](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/CHANGELOG.md)
- The README says it requires MuJoCo ≥ 3.1.3 and was derived from TheRobotStudio's `so101_new_calib.xml`. Derivation steps listed: rounded floats, switched to `implicitfast`, "Use default `forcerange` for actuators rather than 3.35 N/m", added primitive collision geoms for the gripper and arm, added gripper contact parameters "that work well for manipulation", and added a camera mount. — [Menagerie SO101 README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/README.md)
- Global options: `<option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20" impratio="10"/>`. — [so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)
- The actuator class `sts3215` has joint `damping="0.60" frictionloss="0.052" armature="0.028"` and `<position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>`. The XML comment says these gains "are not a 1-to-1 mapping of the servo gains used in Lerobot", were "calculated according to https://github.com/Gregory119/RBE501-RL-arm-project/…", and assume "the servo proportional gain is set to 16". A `backlash` class of ±0.5° is defined but "not added to any joints for now". — [so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)
- Link masses (kg): base 0.147, shoulder 0.100006, upper arm 0.103, lower arm 0.104, wrist 0.079, gripper 0.087, moving jaw 0.012. Joint ranges (rad): shoulder_pan ±1.91986, shoulder_lift ±1.7453, elbow_flex ±1.69 (comment: "5-degree calibration offset applied"), wrist_flex ±1.658, wrist_roll −2.744 to 2.841, gripper −0.1745 to 1.745. — [so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)
- The separate SO-100 model `trs_so_arm100` uses `<position kp="50" dampratio="1" forcerange="-3.5 3.5"/>` and `armature="0.1"`. — [so_arm100.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/trs_so_arm100/so_arm100.xml)

**TheRobotStudio/SO-ARM100 (upstream)**
- `Simulation/SO101/` contains URDFs (`so101_new_calib.urdf`, `so101_old_calib.urdf`, camera variants, and a leader-arm URDF added 2026-09-23) plus MJCFs (`so101_new_calib.xml`, `so101_old_calib.xml`, `scene.xml`, `joints_properties.xml`). The files were generated with onshape-to-robot. "Base collision meshes were removed due to problematic collision behavior". "Motor properties for the STS3215 motors … are adapted from the Open Duck Mini project." There are two calibration conventions: the new one has zero at mid-range, the old one has zero at fully extended. — [SO101 Simulation README](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
- The upstream files disagree with each other. `joints_properties.xml` has `kp="17.8" kv="0.0" forcerange="-3.35 3.35"`. `so101_new_calib.xml` has default-class `kp="998.22" kv="2.731" forcerange="-2.94 2.94"`, but each actuator overrides `forcerange="-3.35 3.35"`. The last actuator-parameter change was "Update actuator model params (#141)" on 2025-11-27. — [joints_properties.xml](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/joints_properties.xml); [so101_new_calib.xml](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.xml)

**Isaac Lab: MuammerBay/isaac_so_arm101**
- The README badges show Isaac Sim 5.1.0, Isaac Lab 2.3.0 and Python 3.11. Tags go up to v1.2.0, the last push was 2026-02-18, and the repo has 318 stars. Install is `uv sync`. "Sim2Real Transfer: Work in progress." — [isaac_so_arm101 README](https://github.com/MuammerBay/isaac_so_arm101)
- Registered tasks: `Isaac-SO-ARM100-Reach-v0`, `Isaac-SO-ARM101-Reach-v0`, `Isaac-SO-ARM100-Lift-Cube-v0` and `Isaac-SO-ARM101-Lift-Cube-v0`, each with a `-Play` variant, trained with RSL-RL PPO. — [tasks/reach/__init__.py](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/reach/__init__.py); [tasks/lift/__init__.py](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/lift/__init__.py)
- `SO_ARM101_CFG` imports the URDF and uses `ImplicitActuatorCfg` for the arm: `effort_limit_sim=1.9`, `velocity_limit_sim=1.5`, stiffness pan/lift/elbow/wrist_flex/wrist_roll = 200/170/120/80/50, damping = 80/65/45/30/20. The gripper has effort 2.5, stiffness 60, damping 20. Other settings: `max_depenetration_velocity=5.0`, `solver_position_iteration_count=8`, `solver_velocity_iteration_count=0`, self-collisions on, and `activate_contact_sensors=False` ("waiting for capsule implementation"). — [so_arm101.py](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/so_arm101.py)
- The URDF masses match Menagerie exactly (0.147 / 0.100006 / 0.103 / 0.104 / 0.079 / 0.087 / 0.012). The URDF joint `<limit effort="10" velocity="10">` values are placeholders, and the joint ranges are the same as Menagerie. — [so_arm101.urdf](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf)
- NVIDIA also has an official learning-path page called "Train a robot arm – SO-101 robot" for Isaac Lab (found in search, not read). — [NVIDIA Isaac Lab learning: 02-so101-robot](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html.md)

**ManiSkill3 (SAPIEN/PhysX)**
- The main branch has `mani_skill/agents/robots/so100/` (uid `so100`, `so100.urdf`) and a `koch` agent. There is no SO-101 agent in the main tree as of 2026-10-30. The latest release is v3.0.1 (2026-04-21). — [ManiSkill repo tree](https://github.com/haosulab/ManiSkill/tree/main/mani_skill/agents/robots/so100); [ManiSkill releases](https://github.com/haosulab/ManiSkill/releases)
- The SO100 agent's `pd_joint_pos` controller uses `stiffness=[1e3]*6`, `damping=[1e2]*6` and `force_limit=100`. The gripper material is static/dynamic friction 2 with restitution 0. — [so_100.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py)
- The robot docs rate it "Quality: A (Values are realistic, but have not been properly identified)". Controllers: `pd_joint_delta_pos`, `pd_joint_pos`, `pd_joint_target_delta_pos`. — [ManiSkill SO100 docs source](https://github.com/haosulab/ManiSkill/blob/main/docs/source/robots/so100/index.md)
- The sim2real digital-twin task `SO100GraspCube-v1` (`max_episode_steps=64`) uses `SimConfig(sim_freq=100, control_freq=20)` and domain randomization of camera, colors and so on. `PickCubeSO100-v1` also exists. — [grasp_cube.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py)

**LeRobot / Genesis / others**
- LeRobot `main` ships simulation env wrappers for LIBERO, MetaWorld, RoboCasa, RoboMME, RoboTwin and VLABench, plus a `hilserl_sim` doc. There is no SO-100/SO-101 simulation env and no SO-101 URDF/MJCF in the repo tree. — [lerobot src/lerobot/envs](https://github.com/huggingface/lerobot/tree/main/src/lerobot/envs)
- A Genesis-format SO101 `scene.xml` exists in the Hugging Face dataset `Genesis-Intelligence/assets` (found in search, not inspected). Genesis's latest GitHub release is v1.4.5 (2026-10-22). — [HF Genesis-Intelligence/assets SO101](https://huggingface.co/datasets/Genesis-Intelligence/assets/blob/main/SO101/scene.xml); [Genesis releases](https://github.com/Genesis-Embodied-AI/Genesis/releases)

### Inferences
- **The sources share one CAD lineage.** Masses, inertias and joint ranges are the same onshape-to-robot export everywhere. They are as good as the CAD density assumptions, which is fine for this task. The paddle (150–190 g) weighs more than the whole wrist + gripper (≈ 0.18 kg), so measuring the paddle's mass and COM matters more than refining the arm's links.
- **The actuator models are the weak point.** Menagerie and Isaac Lab differ by about 5× in stiffness, and their force limits range from 1.9 to 3.35 N·m.
  - Isaac's `velocity_limit_sim=1.5` rad/s will throttle bounce motions. Raise it to match measured servo speed.
  - ManiSkill's `force_limit=100` is about 30× a real STS3215.
  - None of the assets models the Feetech firmware PID, the 1 kHz-class position loop, bus latency or backlash as a delay. For a bounce task, where timing errors of 10–20 ms matter, an identified delay plus first-order servo model will matter more than the choice of simulator.
- **Easiest path per simulator:**
  - MuJoCo/MJX/Warp: start from Menagerie `robotstudio_so101/so101.xml`. Lower the timestep (see §2), add the paddle as a child body of the gripper, and replace the `sts3215` gains with identified values.
  - Isaac Lab 2.3: start from `isaac_so_arm101`'s `SO_ARM101_CFG` with the same edits. It already runs on the user's DGX Spark stack (Isaac Sim 5.1 + Isaac Lab 2.3).

### Gaps
- I did not find a published system identification of the STS3215 (bandwidth, delay, torque–speed curve) on an SO-101. The Menagerie comment points to the RBE501 project as the basis but I did not open it. The STS3215 speed and torque spec was not verified in this research and is probably covered by another research thread.
- The NVIDIA learning-path page and the Genesis SO101 asset were not read in full.
- I could not determine whether any SO-101 asset has measured (rather than CAD-derived) link inertias.

---

## 2. How well does each simulator handle repeated bouncing of a light ball (COR ≈ 0.8–0.9)? Parameter recommendations and pitfalls

### Takeaway
- **MuJoCo (C, MJX and probably Warp)** can produce an accurate, stable COR of about 0.85 for a 2.7 g ball. It needs three things:
  - the direct `solref="-k -b"` format, with b = 2ζ√k and ζ derived from the target e;
  - a timestep of 0.5–1 ms, which is much smaller than Menagerie's 5 ms;
  - k chosen so that √k·dt ≲ 0.5. With dt = 1 ms, k ≈ 1e5 works.
- Measured result: COR is 0.83–0.86 for impact speeds of 1–3 m/s and stays stable over 8 bounces, and MJX matches MuJoCo C within 1%.
- **Three MuJoCo pitfalls**, all measured:
  - positive-format solref is averaged with the paddle's default `0.02 1` (COR collapses to about 0.2);
  - contacts that are too stiff for the timestep gain energy (COR above 1, or explosions);
  - friction rows are too soft by default, so spin transfer comes out about 3× too small. Fix: `solreffriction` on an explicit `<pair>`.
- **PhysX (Isaac Lab, ManiSkill3)** applies restitution as a rigid velocity-level rule. That needs less tuning, but watch three things: the bounce threshold (Isaac Lab default 0.5 m/s), restitution combine modes (default "average" with a paddle at 0.0), and CCD, which runs on CPU only.
- **Genesis** has no documented rigid-rigid restitution coefficient. Its `coup_restitution` applies only to multi-solver coupling and warns of instability.

### Cited Findings
**MuJoCo: documented mechanism**
- solref has "two formats … determined by the sign". In the direct format, "(−stiffness, −damping)" are "scaled by the constraint impedance": b = damping/d_w and k = stiffness·d(r)/d_w². This format "allows direct control over restitution" and is "the recommended format for system identification". In the positive (timeconst, dampratio) format, b = 2/(d_w·timeconst) and k = d(r)/(d_w²·timeconst²·dampratio²). "Smaller [dampratio] values result in under-damped or bouncy constraints." — [MuJoCo docs, Modeling → Solver parameters (source doc/modeling.rst)](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
- The "Restitution" section says: "To achieve perfect restitution … set stiffness to some reasonably large value and damping to zero". Its example is `<geom type="sphere" size="0.1" solref="-1000 0"/>`. Energy is "not exactly preserved because the contact itself is soft and takes several time steps", but it "fluctuates around the initial value instead of drifting". — [MuJoCo docs, Modeling → Restitution](https://mujoco.readthedocs.io/en/latest/modeling.html#restitution)
- "The timeconst parameter should be at least two times larger than the simulation time step". The `refsafe` flag enforces `max(solref[0], 2*timestep)`. — [MuJoCo docs, Solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters); [XML reference, option/flag/refsafe](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-refsafe)
- Mixing rule: with equal priority, solref/solimp are combined by a solmix-weighted average. However, "If solref for either geom is non-positive, i.e., it relies on the direct format, then the element-wise minimum is used regardless of solmix." — [MuJoCo docs, Contact parameters → solref, solimp](https://mujoco.readthedocs.io/en/stable/modeling.html#solmixing)
- A maintainer pointed a "Coefficient of restitution" question (#2080, 2024) to the Restitution docs section. A 2024 report that a ball bounced in MuJoCo but not in MJX with `solref="-1000 0"` (#1668) was confirmed as a bug: "Indeed this is a bug. We'll push out a fix shortly". — [mujoco#2080](https://github.com/google-deepmind/mujoco/issues/2080); [mujoco#1668](https://github.com/google-deepmind/mujoco/issues/1668)
- MuJoCo 3.17.0 was released 2026-10-22. — [MuJoCo releases](https://github.com/google-deepmind/mujoco/releases)

**MuJoCo: my measurements** (ball on a fixed box, target e = 0.85, so ζ = −ln e/√(π² + ln² e) = 0.0517)
- **Direct format, `solref="-k -b"` with b = 2ζ√k, set on the ball only.**
  - At k = 1e5 (b = 32.67): COR = 0.847/0.854/0.859 at 1/2/3 m/s with dt = 0.5 ms; 0.834/0.842/0.850 with dt = 1 ms; 0.805/0.817/0.818 with dt = 2 ms. Peak penetration was 2.9/5.8/8.8 mm and the contact lasted about 11 ms.
  - At k = 1e6, dt = 0.5 ms: COR ≈ 0.86, penetration about 0.9–2.9 mm, contact about 3.5 ms. At dt = 2 ms the same k gave COR 1.3–1.6 (energy gain), and at dt = 5 ms it blew up (COR up to 17).
  - At k = 1e4 and 3 m/s, the ball penetrated 52 mm, more than its own radius, and **passed through the 10 mm paddle**.
  - — [bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py)
- **Positive format `solref="tc 0.0517"` on the ball only** gave COR ≈ 0.13–0.27 at every dt and tc. The paddle's default `0.02 1` was averaged in, giving effective dampratio ≈ 0.53. — [bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py)
- **Positive format set on both geoms (or with `priority`):** at dt = 1 ms, tc = 0.02 → 0.848; tc = 0.01 → 1.54 (energy gain); tc = 0.005 → 8.8. At dt = 2 ms, tc = 0.05 → 0.823 and tc = 0.02 → 1.30. `refsafe` does not prevent this, because the implied ω = 1/(tc·ζ) is far above 1/dt even when tc ≥ 2·dt. — [bounce_test2.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test2.py)
- **Default contact (`solref="0.02 1"`) at dt = 2 or 5 ms: no bounce at all.** — [bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py)
- **Repeated bounces** at k = 1e5, dt = 1 ms gave COR [0.842 0.837 0.839 0.838 0.834 0.857 0.838 0.834]. With b = 0 they were 0.99–1.00, so there is no systematic drift. solimp variations had no measurable effect (all 0.842–0.843). Euler and implicitfast gave 0.842; RK4 gave 0.852. — [bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py)
- **MJX (JAX 0.12.0.1, CPU) vs MuJoCo C**, dt = 1 ms, 2 m/s: direct k = 1e5 gave C 0.842 and MJX 0.852. Positive `0.02 0.0517` on both geoms gave C 0.848 and MJX 0.857. — [bounce_test2.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test2.py)
- **Moving paddle** (mocap, +2 m/s) against the ball at −3 m/s, so 5 m/s relative. The expected ball exit speed is 0.85·5 + 2 = 6.25 m/s.
  - k = 1e5: 6.24/6.23/6.21/5.71 m/s at dt = 0.5/1/2/5 ms.
  - k = 4e5: 6.23/6.21/5.99 m/s, then 18.8 m/s (unstable) at 5 ms.
  - k = 2.5e4: tunneled through the paddle at every dt.
  - — [bounce_test3.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test3.py)
- **Spin transfer (friction).** Setup: shell inertia, 0.5 m/s tangential and 2.4 m/s normal impact. The rigid-body rolling prediction is v_t,out = 0.30 m/s and ω = 15 rad/s.
  - With default friction-row settings, MuJoCo gave v_t,out ≈ 0.44 m/s and ω ≈ 4.2 rad/s, about 30% of the rolling spin. This held for every μ in {0.2, 0.5, 1.0}, pyramidal or elliptic cone, and impratio of 1, 10 or 100.
  - So the soft friction constraint, not μ, limits spin. Slip decays only at the rate set by the normal solref damping (b ≈ 33 s⁻¹).
  - Adding an explicit `<pair … solreffriction="-1 -300"/>` (or `-1 -1000`, or positive `0.002 1`) gave v_t ≈ 0.28–0.29 m/s and ω ≈ 14.5–15.2 rad/s, matching the rolling prediction with normal COR unchanged.
  - `-1 -3000` at dt = 1 ms went unstable (ω = 91 rad/s).
  - In an oblique sliding case (2 m/s tangential, μ = 0.2), the outgoing v_t was 1.27 m/s against a Coulomb prediction of 1.11 m/s. So MuJoCo still under-applies sliding friction by about 15%.
  - — [bounce_test4.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test4.py); [bounce_test5.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test5.py)
- MuJoCo's default sphere inertia is a solid sphere (I/mr² = 0.400). `shellinertia="true"` gives 0.667, the correct value for a thin-walled ping-pong ball. — [XML reference, geom/shellinertia](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-shellinertia); measured in [bounce_test3.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test3.py)

**MuJoCo: aerodynamics**
- The default `fluidcoef` is "0.5 0.25 1.5 1.0 1.0" (blunt drag, slender drag, angular drag, Kutta lift, Magnus lift) for the ellipsoid model. — [XML reference, geom/fluidcoef](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-fluidcoef)
- Measured with ρ = 1.204 and ν = 1.8e-5:
  - Inertia-box model (density only): terminal velocity 6.77 m/s, implied C_d = 0.76.
  - Ellipsoid model with default fluidcoef: 5.91 m/s, implied C_d = 1.00.
  - Ellipsoid with `fluidcoef="0.25 …"`: 8.36 m/s, implied C_d = 0.50.
  - The default ellipsoid Magnus force at v = 5 m/s and ω = 100 rad/s is 10.1 mN, which is 38% of the ball's weight. That is an equivalent C_L ≈ 0.53 at spin ratio 0.4.
  - — [bounce_test2.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test2.py); [bounce_test3.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test3.py)
- MJX-JAX supports only the inertia fluid model ("Fluid model (MJX-JAX): Inertia model only"), and "IMPLICITFAST is not supported with fluid drag". — [MJX docs, feature parity](https://mujoco.readthedocs.io/en/stable/mjx.html)
- DeepMind's 2024 table-tennis system "utilized MuJoCo's ellipsoid-based stateless fluid model". They measured the blunt drag coefficient, kept slender drag at its default, set angular drag to zero, and kept Kutta and Magnus lift at their defaults. — [Achieving Human Level Competitive Robot Table Tennis (arXiv 2408.03906)](https://arxiv.org/html/2408.03906)

**MuJoCo: what MJX supports for this scene**
- In MJX-JAX, "ELLIPSOID and CYLINDER are implemented but only collide with other primitives" and `BOX` is implemented as a mesh. Supported condim values: 1, 3, 4, 6. Supported integrators: EULER, RK4, IMPLICITFAST. MJX-Warp supports all joint types and all solvers except PGS and noslip, but is "not currently" differentiable. — [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)

**PhysX (Isaac Lab 2.3.x, ManiSkill3)**
- Isaac Lab v2.3.2 `PhysxCfg` defaults:
  - `bounce_threshold_velocity = 0.5`: "Relative velocity threshold for contacts to bounce (in m/s)".
  - `enable_ccd = False`.
  - `solver_type = 1` (TGS).
  - `friction_offset_threshold = 0.04`.
  - `friction_correlation_distance = 0.025`.
  - `SimulationCfg.dt = 1/60`.
  - — [IsaacLab v2.3.2 simulation_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sim/simulation_cfg.py)
- The underlying PhysX `PxSceneDesc::bounceThresholdVelocity` is documented as "A contact with a relative velocity below this will not bounce." Its default is "0.2 * PxTolerancesScale::speed". — [PhysX 5.4.1 PxSceneDesc API](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html)
- `RigidBodyMaterialCfg` defaults: static and dynamic friction 0.5, `restitution = 0.0`, both combine modes `"average"`. Note: "When two physics materials with different combine modes collide, the combine mode with the higher priority will be used." Compliant contact (implicit spring) is enabled when `compliant_contact_stiffness > 0`. — [IsaacLab v2.3.2 physics_materials_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sim/spawners/materials/physics_materials_cfg.py)
- In PhysX, compliant contact "is enabled by assigning a negative restitution value to the material". The default solver is "4 position iterations and 1 velocity iteration". TGS offers "Improved handling of high-mass ratios" and "More accurate simulation of high frequency effects". — [PhysX 5.4.1 Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html)
- CCD prevents "tunnelling". Its default is "a finite number of CCD substeps (defaulted to 1) and drops any remaining time". Speculative CCD "can result in tunneling" when the solver moves an actor entirely through objects in one step. The default `contactOffset` is 0.02 m (contact distance 4 cm). — [PhysX 5.4.1 Advanced Collision Detection](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html)
- **On GPU, "Features such as joint projection, CCD and triggers are not GPU accelerated and are still processed on the CPU".** GPU contact generation needs PCM. — [PhysX 5.4.1 GPU Rigid Bodies](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html)
- A forum user reported odd first-bounce heights near restitution 0.2 on GPU PhysX but not on CPU. A moderator attributed it to a bug in which restitution updates did not take effect on GPU, said to be fixed in Isaac Sim ≥ 2022.2.1. — [NVIDIA forum: Restitution has weird behaviour around 0.2](https://forums.developer.nvidia.com/t/restitution-has-weird-behaviour-around-0-2/256411)
- The user's own goalkeeper project already sets `PhysxCfg(bounce_threshold_velocity=0.3, …)` and chooses the restitution combine mode by value ("max" if restitution > 0.3, else "min"). — [goalkeeper_env.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/ppo-goalkeeper/goalkeeper/environments/goalkeeper_env.py); [assets.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/ppo-goalkeeper/goalkeeper/environments/assets.py)

**Genesis**
- `coup_restitution` is described as "Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0". The source warns "Non-zero coup_restitution could lead to instability". It applies to impulse-based coupling between solvers, not rigid-rigid contact. — [Genesis docs, Rigid material](https://genesis-world.readthedocs.io/en/latest/api_reference/material/rigid.html); [Genesis coupling guide](https://genesis-world.readthedocs.io/en/v0.4.3/user_guide/advanced_topics/solvers_and_coupling.html)
- In a manipulation benchmark, Stone Tao (ManiSkill author) observed "the cube falls out of the gripper" in Genesis and called this a sign of accuracy problems. — [How fast is the new hyped Genesis simulator? (Stone Tao, Jan 2025)](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)

**Cross-simulator accuracy benchmark**
- GPUSimBench (arXiv, July 2026) rolled a steel ball into a 3×3×3 cube array and compared simulators with reality by planar EMD at 5 s. Results: ManiSkill 2.52 cm, MuJoCo Playground 3.38, MJX 3.97, MuJoCo Warp 4.23, Genesis 4.78, Isaac Lab 5.40, Madrona 211.6 (XPBD). It did not test restitution or bouncing directly. — [GPUSimBench (arXiv 2607.13059)](https://arxiv.org/html/2607.13059)

### Inferences
**Recommended MuJoCo / MJX / Warp settings**, from the measurements above:
- `timestep` 0.0005–0.001 s. Control at 50–100 Hz means 10–40 physics substeps per policy step.
- Ball geom: `type="sphere" size="0.02" mass="0.0027" shellinertia="true" condim="3"`, plus `solref="-k -b"` set on the ball.
  - k = 1e5 at dt = 1 ms, or k ≈ 4e5–1e6 at dt = 0.5 ms. Keep √k·dt ≤ 0.5.
  - b = 2ζ√k with ζ = 0.0517 for e = 0.85, 0.0708 for e = 0.80, 0.0335 for e = 0.90. For example, k = 1e5 gives b ≈ 32.7 (e = 0.85).
- Because direct-format mixing takes the element-wise minimum, giving the paddle an even more negative solref would silently override the ball's. Simplest: set solref only on the ball, or use an explicit `<pair>` for ball–paddle, ball–floor and ball–arm.
- Add `solreffriction="-1 -300"` to `-1 -1000` on the ball–paddle `<pair>` so spin transfer is realistic.
- Expect roughly 6–15 mm peak penetration at 5 m/s relative speed with k = 1e5. That is acceptable for a 20 mm-radius ball but bad for a thin rim.
- A linear spring-damper gives a nearly speed-independent COR (measured 0.83–0.86 over 1–3 m/s). To get the real speed dependence (see §5/§6), randomize e per episode (e.g. 0.80–0.88) or set solref per env from impact speed. MJX/Warp allow batched per-env model fields; I did not verify this specifically for solref.
- Drag: in MuJoCo C use the ellipsoid model with fluidcoef ≈ "0.25 0.25 0 1 1" for C_d ≈ 0.5. Treat the Magnus coefficient as uncalibrated, since the default looks about 2× high (see §5).
- In MJX-JAX the ellipsoid model is unavailable and implicitfast cannot be combined with fluid forces. Simplest: apply quadratic drag (and optional Magnus) yourself through `xfrc_applied` each substep, and keep implicitfast for the arm.

**Recommended Isaac Lab / PhysX settings** (inferred from the docs; not measured here):
- `sim.dt` = 1/240–1/500 s with decimation to 50–100 Hz.
- `bounce_threshold_velocity` ≈ 0.05–0.2 m/s, so that low dribbles still bounce. The default 0.5 m/s kills any contact slower than that.
- Ball and paddle restitution both ≈ 0.85 with combine mode "max", or explicitly set every surface the ball can hit. The default restitution of 0.0 with "average" would halve the ball's COR.
- PhysX restitution is applied rigidly (no soft contact), so it should be much less dt-sensitive than MuJoCo, but I did not test it here.
- `contact_offset` ≈ 5–10 mm on the ball and paddle, and `rest_offset` = 0.
- The ball's inertia must be set to the shell value (7.2e-7 kg·m²) via the USD mass API. Density-derived inertia gives a solid sphere.
- Add quadratic drag yourself, e.g. with `set_external_force_and_torque` each step. PhysX only has linear and angular damping, which are not quadratic.
- Because CCD runs on CPU only in the GPU pipeline, rely on small dt plus speculative contacts. At ≤ 6 m/s relative speed and dt = 1/240 s, the ball moves 2.5 cm per step, less than its 2 cm radius plus half the paddle thickness plus the contact offset, so tunneling is unlikely.

**Ranking for this task's contact accuracy and tunability:**
1. MuJoCo-family, with the settings above. Verified here, transparent, and identical in C and MJX.
2. PhysX (Isaac Lab or ManiSkill). Rigid restitution with fewer knobs. The bounce-threshold and combine-mode traps are well known, and spin transfer goes through Coulomb friction with patch friction.
3. Genesis. No documented rigid restitution parameter and reported contact-accuracy issues.

### Gaps
- I did not run PhysX/Isaac Lab bounce tests; they need GPU and Isaac Sim. The PhysX COR accuracy for a 2.7 g ball at 1–3 m/s, its dt sensitivity, and the TGS velocity-iteration count needed for restitution are all unverified.
- I did not test MuJoCo Warp; it needs an NVIDIA GPU. Its handling of direct solref and `solreffriction` is assumed to match MuJoCo C, but this is unverified.
- I did not verify whether MJX-JAX honors `solreffriction` on explicit `<pair>`s. My MJX check covered only normal COR.
- The exact PhysX restitution combine-mode priority order (average < min < multiply < max) comes from memory of the PxCombineMode enum and is unverified. Isaac Lab only links to it.
- Genesis rigid-rigid bounce behavior (MuJoCo-like `sol_params`?) is undocumented in what I found.

---

## 3. Alternative: an analytic ball model (ballistic flight with drag and Magnus, plus impulse-based paddle impact) coupled to arm kinematics

### Takeaway
Analytic, impulse-based ball models are the norm in high-performance table-tennis research. Sony AI 2026 uses parametric drag, Magnus and buckling plus residual nets. Gossard 2026 fits racket-bounce models per rubber. DeepMind 2024 used MuJoCo for the ball but had to add special handling for topspin versus underspin. For a single ball bouncing at low speed on a paddle, a hybrid is the most accurate and cheapest option: simulate the arm in MuJoCo/MJX or Isaac, and handle ball flight plus ball–paddle impact analytically in the same JAX or PyTorch step. Its main costs are hand-written contact detection, including the paddle rim and handle, and no ball–arm collisions.

### Cited Findings
- **Sony AI 2026** ("Physics Models for Sim-to-Real Transfer in Professional-Level Robot Table Tennis"; v1 2026-06-27, submitted to RA-L).
  - Aerodynamics: C_D(Re, spin ratio) is piecewise linear, defined at 2.5/7.5/12.5/17.5 m/s. "At 2.5 m/s, C_D ≈ 0.55 across all Sp". The Magnus coefficient is piecewise; at 2.0 m/s it is constant at s = 0.08 up to ω_b = 150 rad/s.
  - Table bounce: e_n = 0.98 + 0.02·v_z⁻ (speed-dependent), μ = 0.25, κ = 3/2, with Lasso residuals.
  - Racket bounce: **e_r = 0.878 − 0.020·‖v_n⁻‖**, tangential e_t = 0.819 − 0.010‖v_T‖, normal spin damping e_s = 0.805, plus a residual neural network.
  - Results: median landing error 0.37 → 0.15 m (−59%), flight RMSE 15.8 → 8.1 mm. Used to train RL policies for a robot "capable of competing against professional players". No physics engine is named.
  - — [arXiv 2606.28805 (HTML)](https://arxiv.org/html/2606.28805); [abstract](https://arxiv.org/abs/2606.28805)
- **DeepMind 2024** (Achieving Human Level Competitive Robot Table Tennis).
  - MuJoCo ellipsoid fluid model with measured blunt drag. Paddle rubber modeled as "two orthogonal passive joints representing a spring-damper system". Ball–rubber parameters were "determined empirically".
  - They found "a bimodal distribution in contact solver parameters for the paddle rubber restitution" between topspin and underspin, with damping ≈ −10 (underspin) versus ≈ −0 (topspin), and switched parameters by incoming spin ("topspin correction").
  - Table and paddle damping and friction were randomized during training.
  - — [arXiv 2408.03906](https://arxiv.org/html/2408.03906)
- **Gossard 2026** ("Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers for Robotic Table Tennis").
  - Dataset: 8,194 bounces, 10 rackets, 2–12 m/s.
  - "Constant-parameter bounce models underperform": mean velocity MAE 33 ± 26 cm/s versus 19 ± 15 cm/s for the GP model. Online adaptation needs about 30 observations.
  - The impulse model "ignores prolonged contact".
  - — [arXiv 2604.11349](https://arxiv.org/html/2604.11349)
- **Sony AI "Bridging the sim2real gap … transformer-based ball states predictor"** (2026) fixes the drag coefficient at 0.4 and the table COR at 0.9 during training (from the search snippet, not read in full). — [arXiv 2606.11464](https://arxiv.org/pdf/2606.11464)
- **JuggleRL** (quadrotor with a racket juggling a ball, 2025/26).
  - Simulated in OmniDrones / Isaac Sim (the paper is inconsistent on which).
  - COR measured by dropping the ball onto a stationary racket: 0.82 in the sweet spot (r = 5.5 cm), 0.64 outside. Domain randomization of COR over [0.75, 0.90].
  - 4,096 envs at 50 Hz, "2 billion simulation frames in under 3 hours" on an RTX 4090.
  - "the racket's uneven surface causes unavoidable deflections".
  - Their ball was 47.2 g, not a ping-pong ball.
  - — [arXiv 2509.24892](https://arxiv.org/html/2509.24892)
- An NII 2026 project trains a robot arm to juggle with RSL-RL PPO (GRU), domain randomization, noise injection and observation history, using ball-height and airtime rewards. This is a slide deck only. — [NII open house F06](https://www.nii.ac.jp/event/upload/oh2026/F06.pdf)

### Inferences
**Pros of the hybrid (analytic ball, simulated arm):**
- The contact is instantaneous, with no 5–10 ms soft contact, so the ball can be stepped at the control rate with exact event-time interpolation. That removes the 10–40× substep cost the ball otherwise needs.
- The literature COR law can be dropped in directly: e(v) = 0.878 − 0.02·v_n, plus tangential restitution and rolling-limited spin.
- Paddle-surface non-uniformity (sweet spot versus rim) and per-episode randomization of every coefficient are easy to add.
- It is the same model family the top table-tennis groups converged on.

**Cons of the hybrid:**
- You must write sphere-versus-oriented-box (or disc) contact, including edge and rim hits, and the handle and arm links are ignored.
- The paddle's reaction to the impulse (2.7 g against ≥ 170 g) has to be added by hand, or ignored at about a 1.5–3% effect.
- Two code paths (the physics engine and the ball model) must stay consistent in frames and timing.

**Pros of full physics (MuJoCo/MJX with direct solref):**
- Everything collides.
- I verified it gives COR 0.83–0.86 and correct moving-paddle exit speeds.

**Cons of full physics:**
- It needs 0.5–1 ms steps.
- COR is roughly speed-independent.
- Spin needs `solreffriction` tuning.
- A soft contact lasts about 10 ms versus about 1 ms in reality (the real duration is unverified; see Gaps), so the servo-driven paddle "pushes through" the contact longer than reality.

**Overall recommendation:** prototype with full MuJoCo/MJX physics (fastest to stand up from Menagerie). If COR speed dependence or spin turns out to matter on the real arm, swap in an analytic ball–paddle impulse.

### Gaps
- I did not find a paper that does exactly an analytic ping-pong ball on a robot-arm-held paddle with RL. The closest are JuggleRL (a quadrotor and a heavier ball) and the NII slide deck.
- The ball–rubber contact duration (often quoted as about 1 ms) was not found in a primary source in this session.
- The i-Sim2Real (2022) and DeepMind 2023 "Robotic Table Tennis" ball models were not reviewed in detail. One search snippet indicates the earlier DeepMind system used PyBullet. — [Robotic Table Tennis project page](https://sites.google.com/view/robotictabletennis) (not opened)

---

## 4. Throughput: env-steps/s for MJX / MuJoCo Playground, MuJoCo Warp, Isaac Lab, ManiSkill3 and Genesis on one GPU

### Takeaway
Published numbers are not apples-to-apples (different tasks, GPUs and timesteps), but all GPU simulators deliver 10⁵–10⁶ steps/s for a single small arm without rendering. The independent GPUSimBench (RTX 5070, Franka random actions) measured Isaac Lab ≈ 1.85M FPS, MJX ≈ 1.91M, MuJoCo Warp ≈ 3.54M, ManiSkill ≈ 149k and Genesis ≈ 11.1M physics steps/s. The ball changes the picture more than the simulator choice does: MuJoCo needs 0.5–1 ms steps, roughly 10–40 substeps per 50–100 Hz policy step, so policy-step throughput falls by that factor. A state-based policy needs no rendering.

### Cited Findings
- **MuJoCo Playground** (arXiv 2502.08844), Table IX, A100, brax PPO, steps/s:
  - PandaPickCube 140,386 ± 1,707; PandaOpenCabinet 136,007; PandaRobotiqPushCube 487,341; LeapCubeReorient 76,354; AlohaSinglePegInsertion 121,119.
  - Locomotion: Go1JoystickFlatTerrain 417,451.
  - LeapCubeReorient training takes about 2,080 s on 1× RTX 4090 and about 670 s on 8× H100 (8,192 envs).
  - Pixel-based Franka with Madrona: about 37,000 steps/s.
  - — [MuJoCo Playground (arXiv 2502.08844)](https://arxiv.org/html/2502.08844)
- **MJX-Warp** steps/s (GPU and batch size not given on the page): Humanoid 3.35M (pure Warp) / 2.67M (JAX FFI `WARP_STAGED`); Aloha Pot 2.45M / 1.96M. — [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)
- **GPUSimBench** (arXiv 2607.13059; RTX 5070 12 GB, physics-only, Δt = 0.01 s). Max envs / FPS / memory:

  | Simulator (version) | Franka scene | Free-fall scene |
  |---|---|---|
  | Isaac Lab (v2.2.1) | 32,768 / ≈1.85M / 8.13 GB | 4,096 / ≈472k |
  | MJX (MuJoCo 3.4.0) | 524,288 / ≈1.91M | 512 / ≈5.3k |
  | MuJoCo Warp | 524,288 / ≈3.54M | 2,048 / ≈166k |
  | ManiSkill (v3.0.0b22) | 32,768 / ≈149k | 4,096 / ≈73.7k |
  | Genesis (v0.3.11) | 131,072 / ≈11.1M | 2,048 / ≈22.3k |
  | MuJoCo Playground (v0.1.0) | 131,072 / ≈135k | n/a |

  — [GPUSimBench (arXiv 2607.13059)](https://arxiv.org/html/2607.13059)
- **Isaac Lab paper** (arXiv 2511.04831): with 8 GPUs and 16,384 envs, "over 900,000 frames per second" for the DextrAH teacher task and "over 1.6 million" for Franka cabinet. A GeForce 5090 workstation "nearly matches a two-GPU RTX Pro 6000 server on the Franka drawer task" because of CPU bottlenecks. — [Isaac Lab (arXiv 2511.04831)](https://arxiv.org/pdf/2511.04831) (via search snippet; not read in full)
- **ManiSkill3**: "up to 30,000+ FPS" with rendering in benchmarked environments. PickCube with state observations reaches about 100% success "after about 1 minute of training" on an RTX 4090. The ManiSkill benchmarking page claims it is faster than Isaac Lab (v1.2.0) "on the majority of settings", with CartPoleBalance as the only state-only task. — [ManiSkill3 paper (arXiv 2410.00425)](https://arxiv.org/abs/2410.00425); [ManiSkill performance benchmarking](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html)
- **Genesis**: the claimed 43M FPS dropped to about 0.29M FPS on a 4090 with corrected benchmark code (Franka), and Genesis was "3-10x slower" than ManiSkill in a manipulation test. This comes from an interested party (the ManiSkill author). A commenter reports the Genesis team's updated benchmark. — [Stone Tao, Jan 2025](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)
- **JuggleRL** (ball juggling, Isaac Sim/OmniDrones): 4,096 envs, 2B frames in under 3 h on an RTX 4090, which is ≥ 185k frames/s at 50 Hz. — [arXiv 2509.24892](https://arxiv.org/html/2509.24892)

### Inferences
- **For this task (6-DoF arm + paddle + 1 free ball, about 1–3 contacts):** physics cost per substep is probably below PandaPickCube's.
  - With dt = 1 ms and 50 Hz control (20 substeps), an estimate is about 50k–300k policy steps/s on one H200 with MJX or Warp. That is enough for 10⁸–10⁹ steps in hours. This is unverified; measure it.
  - Isaac Lab on DGX Spark (GB10, unified memory) will be slower than an H200. I have no GB10 numbers.
  - The analytic-ball hybrid (§3) avoids the substep penalty for the ball. The arm alone could then run at 2–5 ms steps.
- **Collisions are what cost.** MJX's free-fall result (512 envs max, 5.3k FPS) shows how badly MJX-JAX scales with many contacts. Keep contact pairs explicit: ball–paddle, ball–floor, maybe ball–gripper. MJX docs recommend explicit pairs and `max_contact_points`.
- **Hardware fit for the user's cluster:** MJX/Warp run on the x86 H200/L40S nodes, where Isaac Sim does not install, and MJX-JAX also runs on aarch64 GB10. Isaac Lab runs only on GB10.

### Gaps
- No published benchmark of a ball-bouncing or juggling task across simulators exists.
- No published GB10 (DGX Spark) RL throughput numbers were found.
- The Isaac Lab paper's per-task single-GPU table was not read.
- GPUSimBench's free-fall scene composition, and why MJX was limited to 512 envs, were not read in detail.

---

## 5. Physical parameters of a 40 mm, 2.7 g ping-pong ball

### Takeaway
- Mass 2.7 g and diameter 40 mm, giving shell inertia I = (2/3)mr² ≈ 7.2e-7 kg·m².
- ITTF bounce test: a 305 mm drop onto a steel block must rebound 240–260 mm (some versions say 240–265 mm), which implies COR ≈ 0.89–0.93 on a rigid surface.
- C_D ≈ 0.5–0.55 at 2–3 m/s, giving a terminal velocity of about 8–8.4 m/s.
- Drag lowers a 30 cm bounce apex by about 4%, so it is worth modeling.
- Magnus is small at these speeds unless spin exceeds about 50 rad/s.

### Cited Findings
- **ITTF rebound requirement** (sources disagree on the band):
  - 240–260 mm from 305 mm onto a standard steel block, quoting ITTF B.7 "Bounce Conformity". — [PingSkills forum quoting ITTF](https://www.pingskills.com/table-tennis-forum/the-table-tennis-ball-bounce)
  - 240–265 mm for non-celluloid balls (ITTF 2018 document), with the same 240–265 mm in a Bureau of Indian Standards ITTF ball manual. — [PingSkills forum](https://pingskills.com/table-tennis-forum/the-table-tennis-ball); [BIS ITTF ball manual](https://services.bis.gov.in/tmp/cmt_1683725821_M3%20BALL%20MANUAL%20rev5%20BoD2021.pdf)
  - Older sources give 230–250 mm. — [KTH student project](https://www.mech.kth.se/~karlsson/SG1108/Projekt/Studsande%20bordtennisboll.pdf)
- **Drag and Magnus** (Sony AI 2026): ball radius 0.02 m, kinematic viscosity 1.506e-5 m²/s. At 2.5 m/s, C_D ≈ 0.55 for all spin ratios. At 17.5 m/s, C_D is 0.47 (no spin) down to 0.37 (Sp ≈ 0.84). Magnus coefficient at 2.0 m/s is 0.08 up to ω_b = 150 rad/s. "low-speed estimates are noisy but have negligible effect on trajectories". — [arXiv 2606.28805](https://arxiv.org/html/2606.28805)
- Another 2026 Sony AI paper used a fixed C_D = 0.4 in training. — [arXiv 2606.11464](https://arxiv.org/pdf/2606.11464)
- **COR versus impact speed on a hard table:** e_n = 0.98 + 0.02·v_z⁻, with v_z⁻ negative on impact, so e drops as speed rises. — [arXiv 2606.28805](https://arxiv.org/html/2606.28805)
- **Spin at low speed on hard surfaces** (Rémond et al. 2023, glass plate): spin and speed "were both set during the sliding phase" and depend on friction, saturating at the rolling values at small incidence angles. — [APS Physics summary](https://physics.aps.org/articles/v16/s78)

### Inferences (computed here, not cited)
- **Rigid-surface COR from ITTF:** e ≈ √(240/305) = 0.887 up to √(260/305) = 0.923, or 0.932 for 265 mm. This ignores drag, which would push the true COR about 2–3% higher.
- **Terminal velocity:** v_t = √(2mg/(ρAC_D)) = √(35.0/C_D) m/s → 8.4 m/s at C_D = 0.5 and 8.0 m/s at C_D = 0.55. MuJoCo's ellipsoid model with fluidcoef[0] = 0.25 reproduced 8.36 m/s.
- **Drag effect on a 30 cm bounce:** apex height ≈ (v_t²/2g)·ln(1 + v₀²/v_t²) gives about 0.288 m instead of 0.300 m (−4%), and impact speed after a 30 cm fall is about 2% lower. The effect is comparable to COR uncertainty, so include quadratic drag.
- **Reynolds number** at 2 m/s ≈ 5,300.
- **Magnus at bounce speeds:** at v ≈ 2.4 m/s and ω ≈ 15 rad/s (spin ratio about 0.13), with C_L ≈ 0.08 (Sony's low-speed value), the force is about 0.35 mN. That is about 1.3% of weight, or about 1–2 cm of lateral drift per 0.5 s flight. It is second-order next to contact-induced spin (§6), but not zero.
- MuJoCo's default ellipsoid Magnus (C_L ≈ 0.53 at Sp = 0.4) looks several times larger than Sony's low-speed coefficient. Either set the last fluidcoef term (Magnus) to about 0.1–0.2, or disable it and add your own. The definitions of the coefficients differ, so this is unverified.

### Gaps
- I could not retrieve the current ITTF technical leaflet T3 directly, so the exact rebound band (240–260 versus 240–265 mm) is unconfirmed.
- No primary measurement of C_D for a 40 mm plastic (ABS) ball below 2 m/s was found. Sony notes low-speed estimates are noisy.
- Plastic (40+) versus celluloid ball differences were not researched.

---

## 6. (User update) A regular rubber-covered paddle (150–190 g) held in the gripper: COR and friction at 1–2 m/s, speed dependence, how to model it, and whether spin matters

### Takeaway
- **COR on inverted rubber:** about 0.84–0.86 at 1–2 m/s (Sony AI: e = 0.878 − 0.020·v_n), falling about 0.02 per m/s, with similar slopes across studies (0.017–0.021).
- **Bare wood:** I found no primary data.
- **Friction:** the fitted Coulomb μ for a rubber bounce model is about 0.2. But at near-normal bounce angles the ball reaches rolling anyway, so the exact μ hardly matters. What matters is the inertia ratio and the rubber's tangential springback (e_t ≈ 0.82).
- **Spin:** contact-induced spin does matter at these low speeds. A 0.5 m/s tangential slip leaves the paddle with about 15 rad/s, and that spin kicks the next bounce sideways. Aerodynamic Magnus is minor.
- **Modeling:** a 15–16 cm disc or box (about 14 mm thick) plus a handle box, with the paddle's measured mass and COM. In MuJoCo use direct solref with k ≈ 1e5 and b ≈ 33 at dt = 1 ms, plus `solreffriction="-1 -300"`. Randomize e over about 0.80–0.88.

### Cited Findings
- **Racket (rubber) COR model:** e_r = 0.878 − 0.020·‖v_n⁻‖, tangential e_t = 0.819 − 0.010‖v_T‖, normal spin damping e_s = 0.805, grip fraction α = (1 + e_t)/(1 + κ) with κ = 3/2 (thin shell). — [Sony AI 2026, arXiv 2606.28805](https://arxiv.org/html/2606.28805)
- **Gossard 2026** (andro Power 3 allround and Hexer Duro offensive inverted rubbers, among others).
  - For inverted rubbers, COR "decreases approximately linearly with impact speed |v_z|", by up to 0.15 between about 2 and 12 m/s. Fitted slopes average 0.021 (offensive) and 0.017 (allround) per m/s. Intercepts were not reported in the text I read.
  - Anti-spin rubber has COR = 0.53. Pips behave nonlinearly and depend on tangential speed.
  - Friction μ = 0.197 (smooth-transition model) versus 0.207 (standard Coulomb), with a rolling plateau at α = 0.4. For inverted rubbers the tangential parameter k_p is about 0.019.
  - No bare-blade test.
  - — [arXiv 2604.11349](https://arxiv.org/html/2604.11349)
- **ISJOS study** (table-tennis ball on a rubber paddle, 1.4–20.2 m/s): COR "decreased linearly with increasing impact velocity", with a slope of about 0.018 per 1 m/s (from the search snippet; the PDF is image-only and I could not read it). — [ISJOS v15 p2](https://www.isjos.org/pdfs/ISJOS_v15_p2.pdf)
- **Finite-element and experimental study of ball/rubber impact:** "the restitution coefficients are seen to decrease with increasing launching velocity, while being minimum when the two racket polymeric constituents are associated", i.e. sponge plus topsheet together give the lowest COR. — [Non Linearity of the Ball/Rubber Impact in Table Tennis (Procedia Eng. 2016)](https://www.sciencedirect.com/science/article/pii/S1877705816307548)
- A 2026 comparison of four rubber types (pips-out, raw, long, inverted) found a linear incident-velocity versus COR correlation (R² > 0.84). — [Jia et al. 2026, Proc. IMechE Part P](https://journals.sagepub.com/doi/10.1177/17543371241310338) (abstract-level)
- **Paddle non-uniformity:** in JuggleRL (a different ball and racket), COR was 0.82 in the central 5.5 cm-radius sweet spot versus 0.64 near the edge. — [arXiv 2509.24892](https://arxiv.org/html/2509.24892)
- DeepMind modeled the paddle rubber's compliance as two orthogonal passive spring-damper joints, and needed spin-dependent rubber damping. — [arXiv 2408.03906](https://arxiv.org/html/2408.03906)
- **Spin generation measured in MuJoCo** (shell ball, 2.4 m/s normal, 0.5 m/s tangential):
  - Default friction rows: ω ≈ 4 rad/s.
  - With `solreffriction="-1 -300"`: ω ≈ 14.5–15 rad/s, matching the rigid rolling prediction of 15 rad/s.
  - μ ∈ {0.2, 1.0} made no difference in this near-normal regime.
  - — [bounce_test4.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test4.py); [bounce_test5.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test5.py)

### Inferences
**COR at the user's speeds:**
- Sony's law gives 0.858 at 1 m/s and 0.838 at 2 m/s. Gossard's and ISJOS's slopes (0.017–0.021 per m/s) agree with it.
- Speed dependence over 1–2 m/s is therefore only about 0.02 in e. That is smaller than paddle-to-paddle and sweet-spot-versus-rim variation (JuggleRL: 0.82 versus 0.64 on their racket).
- **Domain-randomizing e in about [0.80, 0.88] per episode** (or per contact location) is more important than modeling e(v).
- Measure your own paddle: drop the ball from 30 cm onto the paddle clamped in the gripper, film at ≥ 120 fps, and compute e = √(h_r/h_d) corrected for drag. Repeat at the center and near the rim.

**Bare wood:** no data was found. Physically, a hard wood blade should behave closer to the ITTF steel-block COR (about 0.89–0.92) than rubber does, with lower friction. This is unverified, and irrelevant if both sides of the paddle are rubbered.

**Friction and spin at 1–2 m/s:**
- Using Sony's rolling switch α = min(0.4, μ(1 + e)|v_z|/|v_T|) with μ ≈ 0.2 and e ≈ 0.85, the ball reaches rolling whenever |v_T| ≲ 0.93|v_z|, i.e. for incidence up to about 43° from the paddle normal. Bounce-juggling stays inside this regime, so the outgoing spin is set by rolling (ω ≈ 0.6·v_T/r for a shell), not by μ.
- Inverted rubber's positive tangential restitution (e_t ≈ 0.8) means it can even reverse the contact-point slip. This "springy rubber" effect cannot be produced by MuJoCo or PhysX Coulomb friction. It would need DeepMind-style passive tangential joints on the paddle surface, or an analytic impulse model.
- Spin carries over: a ball leaving with ω ≈ 15 rad/s arrives at the next impact with r·ω ≈ 0.3 m/s of contact-point slip even if the paddle is perfectly level. That produces a lateral velocity kick of about 0.1–0.2 m/s, comparable to the corrections the policy makes.
- **So model contact spin correctly:** use shell inertia, `solreffriction` tuned as above, and include ball angular velocity in the observation or let the policy infer it from history. Aerodynamic Magnus at these speeds is about 1% of weight and second-order.
- On the real arm, angular velocity is hard to observe. A recurrent or history-based policy (as in the NII juggling work) is a reasonable hedge.

**Paddle geometry for the simulator:**
- Typical blade head ≈ 157 × 150 mm. Use the user's 150–190 g total and measure the actual paddle, since these dimensions come from general knowledge, not a cited source.
- **Head:** a cylinder of radius ≈ 0.076–0.078 m and half-thickness ≈ 0.007 m in MuJoCo (sphere–cylinder is a supported primitive pair in MJX-JAX), or a box of 0.157 × 0.150 × 0.014 m in PhysX, where boxes are native. Mass ≈ 0.12–0.14 kg. For a box the inertia is about Ixx ≈ m(b² + t²)/12 ≈ 2.5e-4, Iyy ≈ 2.7e-4, Izz ≈ 5.1e-4 kg·m² at 0.13 kg. Easier: give each geom a `mass` and let the simulator compute the inertia.
- **Handle:** a box of about 0.10 × 0.028 × 0.025 m, ≈ 0.03–0.05 kg, rigidly welded inside the gripper at the measured grasp offset. Make it a child body of `gripper` (MuJoCo) or a fixed-joint link (URDF/USD).
- Disable handle–gripper self-collision.
- **Contact parameters** (ball–paddle, MuJoCo at dt = 1 ms):
  `<pair geom1="ball" geom2="paddle_head" condim="3" friction="0.3 0.3 0.005 1e-4 1e-4" solref="-1e5 -32.7" solreffriction="-1 -300"/>`.
  With e = 0.85 the policy will see COR of about 0.83–0.85 (measured). Randomize b from about 20 to 50 to cover e of about 0.80 to 0.90 at k = 1e5 (estimated from ζ).
- **Contact parameters** (PhysX): ball and paddle restitution 0.85 with combine "max", static and dynamic friction about 0.3–1.0, `bounce_threshold_velocity` 0.05–0.2 m/s, and dt ≤ 1/240 s.

**Load on the SO-101:** a 150–190 g paddle with its COM roughly 8–12 cm beyond the grip adds about 0.15–0.25 N·m of static wrist torque. That is far more than the 2.7 g ball's impulse (≈ 0.012 N·s at 2.4 m/s), and a large share of the 1.9–2.94 N·m modeled servo limit when the arm is extended. Put the paddle's true mass and COM in the simulator, and use realistic servo force limits rather than ManiSkill's 100 N·m. Arm torque limits are better verified by the hardware research thread.

**Paddle recoil:** with m_ball/m_paddle ≈ 0.016, a free paddle's recoil would lower apparent COR by about 1.5–3%. A servo-held paddle sits between free and rigid. Published racket CORs were measured with real rackets, so they already include this. In simulation, attach the paddle with its real mass rather than as a fixed or mocap body.

### Gaps
- I found no published COR or friction values for a ping-pong ball on a bare wooden blade at 1–2 m/s.
- I found no published intercepts for Gossard's inverted-rubber COR fits at below 2 m/s (their dataset starts at about 2 m/s). Sony's e_r law is fitted on competitive-match data and is extrapolated to 1 m/s here.
- No direct measurement of the static or kinetic friction coefficient of the ball on inverted rubber was found (rubbers are tacky; μ is likely > 0.5). The fitted μ ≈ 0.2 is an effective bounce-model parameter, not a tribometer value.
- I did not test MuJoCo's ability to emulate rubber tangential springback (e_t > 0), e.g. via a passive tangential joint as DeepMind did.
- Paddle dimensions and masses are typical values, not from a cited source. Measure the user's actual paddle.
