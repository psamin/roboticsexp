#!/bin/bash
# Run the trained policy on the SO-101. The model runs on a Futurama GPU (never on this Mac);
# this Mac only drives the arm and cameras (LeRobot async inference).
#
#   1. On the cluster:  MODEL=<model dir> sbatch ~/serve_policy.sbatch   (or ./run_policy.sh prints it)
#   2. Here:            ./run_policy.sh        Ctrl-C stops the arm (support it first: torque turns off)
#
# Needs GT VPN. Override the model with MODEL=/nethome-instruction/pi34/models/<dir> ./run_policy.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOGIN=pi34@planetexpress-login.cc.gatech.edu
PORT=8080
mkdir -p "$SO101/logs"
LOG=$SO101/logs/policy-$(date +%Y%m%d-%H%M%S).log

# The policy server overrides the camera-name mapping saved at training time with an empty one
# (policy_server.py, rename_observations_processor), so name the cameras exactly as the model
# saw them in training: front -> camera1, wrist -> camera2. Same devices and settings as recording.
POLICY_CAMERAS="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}, camera2: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}}"

MODEL=${MODEL:-$(ssh -o BatchMode=yes $LOGIN 'ls -td /nethome/pi34/models/smolvla-* 2>/dev/null | head -1' 2>/dev/null | sed 's|^/nethome/|/nethome-instruction/|')}
[ -n "$MODEL" ] || { echo "No trained model found in ~/models on the cluster." >&2; exit 1; }
echo "Model: $MODEL"

JOB=$(ssh -o BatchMode=yes $LOGIN "squeue -u pi34 -n smolvla-serve -t RUNNING -h -o %i | head -1" 2>/dev/null)
if [ -z "$JOB" ]; then
  echo "No policy server running. Start one on the cluster (1x H200, 2 h), then rerun this:" >&2
  echo "  ssh $LOGIN 'MODEL=$MODEL sbatch ~/serve_policy.sbatch'" >&2
  exit 1
fi
NODE=$(ssh -o BatchMode=yes $LOGIN "scontrol show node \$(squeue -j $JOB -h -o %N) | grep -oE 'NodeHostName=\\S+' | cut -d= -f2")
echo "Policy server: job $JOB on $NODE. Waiting for it to finish setting up ..."
for _ in $(seq 60); do
  ssh -o BatchMode=yes $LOGIN "grep -q SERVING /nethome/pi34/logs/smolvla-serve-$JOB.log" 2>/dev/null && break
  sleep 5
done
sleep 5  # the server starts listening right after it prints SERVING

# Tunnel: Mac localhost:$PORT -> the compute node's localhost:$PORT, through the login node.
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -J $LOGIN -L $PORT:127.0.0.1:$PORT pi34@$NODE &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null || true' EXIT
sleep 3
kill -0 $TUNNEL 2>/dev/null || { echo "SSH tunnel to $NODE failed." >&2; exit 1; }

start_camera_lock
# max_relative_target caps each joint's move per control step (degrees): a safety limit for first runs.
"$PY" -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:$PORT \
  "${FOLLOWER_ARGS[@]}" "--robot.cameras=$POLICY_CAMERAS" --robot.max_relative_target=${MAX_STEP:-10} \
  --task="$TASK" --policy_type=smolvla --pretrained_name_or_path="$MODEL" --policy_device=cuda \
  --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30 \
  2>&1 | tee "$LOG"
