#!/bin/bash
# E6c: the SIMSAT2D4F full81 training (started by chain_e6.sh) is staged only if the E8 matched-emphasis held check passes.
cd "$(dirname "$0")/../.."
D=runs/SIMSAT2D4F_full81
until test -f $D/training_candidate/report.json || ! ps -eo args= | grep -q "train_simati_focus.py --run $D\$"; do sleep 60; done
test -f $D/training_candidate/report.json || { echo "SIMSAT2D4F_full81 FAILED (training ended without report)"; exit 4; }
echo "$(date -u +%H:%M) SIMSAT2D4F_full81 trained; waiting for the held verdict"
until grep -qE "SIMSAT2D4M VERDICT|SIMSAT2D4M .*failed" runs/TTAKD/e8_match.log 2>/dev/null; do sleep 60; done
grep -q "SIMSAT2D4M VERDICT PASS" runs/TTAKD/e8_match.log || { echo "E6 not staged (E8 matched-emphasis check): $(grep SIMSAT2D4M runs/TTAKD/e8_match.log | tail -1)"; exit 0; }
.venv310/bin/python runs/CYCLE4/finish_generic.py $D SIMSAT2D4FH full81_SIMAH_r1 > $D/finish.log 2>&1
grep -q "timing exit 0" $D/finish.log || { echo "SIMSAT2D4FH gates failed: $(grep exit $D/finish.log | tr '\n' ' ')"; exit 5; }
.venv310/bin/python runs/EXACTLANE/make_fixed5.py $D F5 > runs/EXACTLANE/SIMSAT2D4FHF5_build.log 2>&1
grep -q "timing exit 0" runs/EXACTLANE/SIMSAT2D4FHF5_build.log || { echo "SIMSAT2D4FHF5 gates failed"; exit 6; }
.venv310/bin/python runs/EXACTLANE/make_malloc.py SIMSAT2D4FHF5 MA > runs/EXACTLANE/SIMSAT2D4FHF5MA_build.log 2>&1
echo "$(date -u +%H:%M) E6 chain done: $(grep exit runs/EXACTLANE/SIMSAT2D4FHF5MA_build.log | tr '\n' ' ')"
runs/TTAKD/stage_d4.sh SIMSAT2D4F SIMSAT2D4M
.venv310/bin/python runs/NEXTSLOT_20260924/anchor_forecast.py SIMSAT2D4FHF5MA runs/TTAKD/result_SIMSAT2D4M.json --amend
