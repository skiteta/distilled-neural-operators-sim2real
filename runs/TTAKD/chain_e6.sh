#!/bin/bash
# E6: full81 4-block student of SIMSAT2H with focused simulation replay started at once (speculative); after training, wait for the
# held verdict (rule_SIMSAT2D4F.json): PASS -> gates -> F5 -> MA -> staging -> anchored forecast; otherwise stop.
cd "$(dirname "$0")/../.."
.venv310/bin/python - <<'PY'
import json,shutil
from pathlib import Path
R=Path.cwd();S=R/'runs/SIMSAT2D4_full81';D=R/'runs/SIMSAT2D4F_full81';D.mkdir(exist_ok=False)
p=json.loads((S/'plan.json').read_text());p['candidate']='SIMSAT2D4FH';p['runtime']='runs/SIMSAT2D4F_full81/teacher_runtime'
p['training'].update(sim_focus_fraction=0.5,stream_seed_offset=1800);p.pop('simsat2d4',None)
p['simsat2d4f']='full81 4-block student of SIMSAT2H with focused simulation replay (4 of 8 sim windows from the 18 regimes without released real data), SIMSAT2D4_full81 recipe otherwise, real stream 1800'
(D/'plan.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n');(D/'training_script.txt').write_text('runs/TTAKD/train_simati_focus.py\nruns/SIMALL/sim_replay_focus.py\n')
shutil.copytree(S/'teacher_runtime',D/'teacher_runtime');print('prepared SIMSAT2D4F_full81',flush=True)
PY
D=runs/SIMSAT2D4F_full81
for try in 1 2 3 4; do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 3500 ]; do sleep 5; done
  test -d $D/training_candidate && mv $D/training_candidate $D/training_candidate_CRASH_$try && mv $D/train.log $D/train_CRASH_$try.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup .venv310/bin/python runs/TTAKD/train_simati_focus.py --run $D > $D/train.log 2>&1 &
  echo "$(date -u +%H:%M) SIMSAT2D4F_full81 launched (try $try)"
  until test -f $D/training_candidate/report.json || ! ps -eo args= | grep -q "train_simati_focus.py --run $D\$"; do sleep 60; done
  test -f $D/training_candidate/report.json && break
  echo "$(date -u +%H:%M) SIMSAT2D4F_full81 crashed: $(grep -m1 -E 'Error' $D/train.log | cut -c1-100)"
done
test -f $D/training_candidate/report.json || { echo "SIMSAT2D4F_full81 FAILED"; exit 4; }
echo "$(date -u +%H:%M) SIMSAT2D4F_full81 trained; waiting for the held verdict"
until grep -qE "SIMSAT2D4F VERDICT|SIMSAT2D4F .*failed" runs/TTAKD/e6_focus.log 2>/dev/null; do sleep 60; done
grep -q "SIMSAT2D4F VERDICT PASS" runs/TTAKD/e6_focus.log || { echo "E6 not staged: $(grep SIMSAT2D4F runs/TTAKD/e6_focus.log | tail -1)"; exit 0; }
.venv310/bin/python runs/CYCLE4/finish_generic.py $D SIMSAT2D4FH full81_SIMAH_r1 > $D/finish.log 2>&1
grep -q "timing exit 0" $D/finish.log || { echo "SIMSAT2D4FH gates failed: $(grep exit $D/finish.log | tr '\n' ' ')"; exit 5; }
.venv310/bin/python runs/EXACTLANE/make_fixed5.py $D F5 > runs/EXACTLANE/SIMSAT2D4FHF5_build.log 2>&1
grep -q "timing exit 0" runs/EXACTLANE/SIMSAT2D4FHF5_build.log || { echo "SIMSAT2D4FHF5 gates failed"; exit 6; }
.venv310/bin/python runs/EXACTLANE/make_malloc.py SIMSAT2D4FHF5 MA > runs/EXACTLANE/SIMSAT2D4FHF5MA_build.log 2>&1
echo "$(date -u +%H:%M) E6 chain done: $(grep exit runs/EXACTLANE/SIMSAT2D4FHF5MA_build.log | tr '\n' ' ')"
runs/TTAKD/stage_d4.sh SIMSAT2D4F
.venv310/bin/python runs/NEXTSLOT_20260924/anchor_forecast.py SIMSAT2D4FHF5MA runs/TTAKD/result_SIMSAT2D4F.json --amend
