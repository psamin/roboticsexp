# blocks_ranking_rgb findings

Rounds used: 4/4 (8 run-tool calls). Final: clean 5/5, randomized 5/5.
check_success: red.x < green.x < blue.x, adjacent |dx|<0.13, |dy|<0.03, both grippers open.

| Round | clean | rand | change |
|---|---|---|---|
| r1 (v1) | 4/5 | - | colour-cloud perception, near-arm pick, row of 3 slots |
| r2 | - | 2/5 | perceive once in setup |
| r3 | 5/5 | 4/5 | footprint/height gate on colour blobs; aperture-verified grasp retry |
| r4 | 5/5 | 5/5 | verify end-pose reached the grasp target, else next yaw/approach |

## Per-seed root causes
- 103 (r1): re-perceiving blue mid-episode failed, the arm occluded the head camera. Fix: perceive all three in setup.
- 152 (blue), 154 (red) (r2): same-colour clutter (half-extent 0.03, ~1600 points) was the largest colour blob. Fix: bound blob footprint (<0.075) and height (<0.055).
- 151 (r3): right arm's planner returned success for the blue grasp but ended 5.8 cm short (jaws closed on air, aperture 0.002, twice). Fix: compare get_observation endpose with target, ladder yaw/approach if >3 cm off. r4 flipped it.
- 153, 155 passed throughout after r2.

## Fitted to debug scenes
Nothing known. Constants are robot properties (lift 0.12, approach 0.10, jaw gap 0.06, reach tolerance 0.03) plus blob bounds for 3-5 cm cubes. Colour thresholds are generic. Row y = mean spawn y - 0.17 clamped to [-0.20,-0.10] (reach band, derived from the spawn band). Slot x spacing from perceived block size, clamped 0.09-0.12.

## Clean-room attestation
Read only the brief's listed files (subagent prompt sections, envs/blocks_ranking_rgb.py, arm/gripper surfaces, the four skill files and program-contract). No frozen programs, tasks_indep, ext, or ssh. No git, no paid APIs. Only seeds 101-105, 151-155.

Distinct scenes: 10 (perceived poses differ per seed). Frames/episode, cycle counts and play_once comparison not measured (skipped per brief).
