"""Watch a running Phase B training log for problems."""
from __future__ import annotations

import re
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training_unified_phaseB_gelu_av.log")
CHECK_INTERVAL = 30       # seconds between checks
STAGNATION_WINDOW = 5000  # steps
STAGNATION_TOL = 0.95     # median hasn't improved by this factor
SPIKE_THRESHOLD = 100.0   # PDE_B above this is a spike
SPIKE_RATE_MAX = 0.05     # max acceptable spike rate (fraction of steps)

def parse_losses(lines: list[str]) -> dict:
    """Extract step numbers and loss values from log lines."""
    steps, pde, data, rh, bc = [], [], [], [], []
    for line in lines:
        m = re.search(r"step=\s*(\d+).*?PDE_B=([\d.e+\-]+)", line)
        if not m:
            continue
        step = int(m.group(1))
        pde_val = float(m.group(2))
        dm = re.search(r"data_B=([\d.e+\-]+)", line)
        rm = re.search(r"RH_B=([\d.e+\-]+)", line)
        bm = re.search(r"BC_B_outflow=([\d.e+\-]+)", line)
        steps.append(step)
        pde.append(pde_val)
        if dm: data.append(float(dm.group(1)))
        if rm: rh.append(float(rm.group(1)))
        if bm: bc.append(float(bm.group(1)))
    return {"step": steps, "pde": pde, "data": data, "rh": rh, "bc": bc}


def check_nan(lines: list[str]) -> bool:
    """Check for NaN/inf in recent lines."""
    tail = "".join(lines[-50:])
    return "nan" in tail.lower() or "inf" in tail.lower()


def check_spikes(pde: list[float], steps: list[int]) -> tuple[int, float]:
    """Count spikes and spike rate in recent window."""
    if not pde:
        return 0, 0.0
    spikes = sum(1 for v in pde if v > SPIKE_THRESHOLD)
    rate = spikes / len(pde) if pde else 0
    return spikes, rate


def check_stagnation(pde: list[float]) -> bool:
    """True if PDE_B hasn't improved meaningfully."""
    if len(pde) < 100:
        return False
    first_half = pde[:len(pde)//2]
    second_half = pde[len(pde)//2:]
    med_first = statistics.median(first_half)
    med_second = statistics.median(second_half)
    # Improved means lower
    return med_second > med_first * STAGNATION_TOL


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    print(f"[{now()}] watching {LOG} every {CHECK_INTERVAL}s")
    last_line_count = 0
    stale_checks = 0

    while True:
        if not LOG.exists():
            print(f"[{now()}]  log not found, waiting...")
            time.sleep(CHECK_INTERVAL)
            continue

        with open(LOG, errors="replace") as f:
            lines = f.readlines()

        # Only look at lines we haven't seen
        if len(lines) == last_line_count:
            stale_checks += 1
            if stale_checks > 20:
                print(f"[{now()}] !! training may have stopped (no new lines for {stale_checks*CHECK_INTERVAL}s)")
                stale_checks = 0
            time.sleep(CHECK_INTERVAL)
            continue
        stale_checks = 0
        last_line_count = len(lines)

        d = parse_losses(lines)
        if not d["step"]:
            time.sleep(CHECK_INTERVAL)
            continue

        last_step = d["step"][-1]
        recent_n = 200

        # --- NaN check ---
        if check_nan(lines):
            print(f"[{now()}] !! NaN/inf detected at step {last_step} — STOP AND FIX: check PDE sampling, reduce lr, or increase av_ell")
            break

        # --- Stage detection ---
        stage_line = [l for l in lines if "losses=" in l]
        current_stage = stage_line[-1].strip() if stage_line else "?"

        # --- Recent stats ---
        pde_recent = d["pde"][-recent_n:]
        data_recent = d["data"][-recent_n:] if d["data"] else []
        rh_recent = d["rh"][-recent_n:] if d["rh"] else []

        if not pde_recent:
            time.sleep(CHECK_INTERVAL)
            continue

        pde_med = statistics.median(pde_recent)
        pde_min = min(pde_recent)
        data_med = statistics.median(data_recent) if data_recent else 0
        rh_med = statistics.median(rh_recent) if rh_recent else 0
        spikes, spike_rate = check_spikes(pde_recent, d["step"][-recent_n:])

        # --- Status line ---
        status = f"[{now()}] step={last_step}  PDE_med={pde_med:.4f}  PDE_min={pde_min:.4f}  data_med={data_med:.2f}  RH_med={rh_med:.2f}  spikes={spikes}  [{current_stage}]"
        print(status)

        # --- Spike check ---
        if spike_rate > SPIKE_RATE_MAX:
            print(f"[{now()}] !! spike rate {spike_rate:.0%} > {SPIKE_RATE_MAX:.0%} — consider increasing av_ell or reducing lr")

        # --- Stagnation check (only after 10000+ steps in B2) ---
        if last_step > 10000 and "B2" in current_stage:
            if check_stagnation(pde_recent):
                print(f"[{now()}] !! PDE_B stagnating — consider reducing lr or increasing data_B weight")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
