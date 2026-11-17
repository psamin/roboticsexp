#!/bin/bash
# Client for cluster/trial_server.sbatch: replay one program on RoboTwin debug seeds and fetch the results.
#   aw0-aspire/rt.sh <task> <program.py> <tag> <seed>...
# Prints result.tsv (seed, config, trial dir with reward, seconds, check_success exceptions) and the local
# folder holding per-seed logs, summaries and frames. Refuses the locked validation band (201-270).
set -euo pipefail
LOGIN=pi34@planetexpress-login.cc.gatech.edu
SSH=(ssh -o ConnectTimeout=20 -o BatchMode=yes "$LOGIN")
task=$1; prog=$2; tag=$3; shift 3; seeds="$*"
for s in $seeds; do [ "$s" -ge 201 ] && [ "$s" -le 270 ] && { echo "seed $s is in the locked band 201-270" >&2; exit 2; }; done
req="${task}-${tag}-$(date +%H%M%S)"
local_dir=${RT_LOCAL:-$HOME/.cache/aw0-aspire-rt}/$req
"${SSH[@]}" "test -f ~/aw0-aspire-queue/SERVER_READY" || { echo "trial server is not running" >&2; exit 3; }
"${SSH[@]}" "mkdir -p ~/aw0-aspire-queue/staging/$req"
scp -q -o BatchMode=yes "$prog" "$LOGIN:aw0-aspire-queue/staging/$req/program.py"
printf 'TASK=%q\nSEEDS=%q\n' "$task" "$seeds" | "${SSH[@]}" "cat > ~/aw0-aspire-queue/staging/$req/req.env && mv ~/aw0-aspire-queue/staging/$req ~/aw0-aspire-queue/inbox/"
echo "queued $req" >&2
until "${SSH[@]}" "test -f ~/aw0-aspire-queue/out/$req/DONE" 2>/dev/null; do sleep 20; done
mkdir -p "$local_dir"
scp -q -r -o BatchMode=yes "$LOGIN:aw0-aspire-queue/out/$req/." "$local_dir/"
cat "$local_dir/result.tsv"
echo "results: $local_dir"
