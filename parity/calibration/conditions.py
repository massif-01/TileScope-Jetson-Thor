"""Construct and verify an optional power/clock change and restoration plan without writing device state."""
import argparse
import json
from pathlib import Path
import re

try:
    from .run import conditions,read,write
except ImportError:
    from run import conditions,read,write


def inventory():
    clock=Path("/sys/class/devfreq/gpu-gpc-0")
    config=Path("/etc/nvpmodel.conf")
    modes=re.findall(r"<\s*POWER_MODEL\s+ID\s*=\s*(\d+)\s+NAME\s*=\s*([^\s>]+)",read(config) or "")
    return {"measured":conditions(),"power_mode_file":str(config.resolve()),
            "mode_names":dict(modes),"frequency_source":str(clock/"available_frequencies"),
            "frequencies_hz":[int(x) for x in (read(clock/"available_frequencies") or "").split()],
            "governor_source":str(clock/"available_governors"),"governors":(read(clock/"available_governors") or "").split()}


def proposal(inv, mode, hz, governor):
    actual=inv["measured"]
    prior={"mode_id":actual["mode_id"],"governor":actual["clock"]["governor"],
           "min_hz":int(actual["clock"]["min_freq"]),"max_hz":int(actual["clock"]["max_freq"])}
    if any(str(m) not in inv["mode_names"] for m in (prior["mode_id"],mode)):
        raise ValueError("unlisted or unreadable power mode")
    if hz not in inv["frequencies_hz"] or governor not in inv["governors"] or prior["governor"] not in inv["governors"]:
        raise ValueError("unlisted GPU frequency or governor")
    target=dict(mode_id=mode,governor=governor,min_hz=hz,max_hz=hz)
    # Power profiles can reset governor/limits. Both procedures re-read after mode selection.
    protocol=["select declared power profile if needed", "read new limits and re-check availability under selected profile",
              "if target minimum exceeds current maximum, raise maximum first; otherwise write minimum then maximum",
              "set target governor", "read back mode, governor, min and max; compare with exact target"]
    # Verify the range order in the two directions, including disjoint ranges.
    def steps(start,goal):
        lo,hi=start["min_hz"],start["max_hz"]
        bounds=[("max_hz",goal["max_hz"]),("min_hz",goal["min_hz"])] if goal["min_hz"]>hi else [
            ("min_hz",goal["min_hz"]),("max_hz",goal["max_hz"])]
        cur=dict(start,mode_id=goal["mode_id"])
        for key,v in bounds:
            cur[key]=v
            if cur["min_hz"]>cur["max_hz"]:
                raise ValueError("intermediate inverted frequency bounds")
        cur["governor"]=goal["governor"]
        if cur!=goal:
            raise ValueError("incomplete restoration")
        return bounds
    return dict(kind="dry_run",inventory=inv,hardware_changes_executed=False,apply_target=target,
                restore_target=prior,procedure=protocol,apply_bound_order=steps(prior,target),
                restore_bound_order=steps(target,prior),
                readback="nvpmodel -q; /sys/class/devfreq/gpu-gpc-0/{governor,min_freq,max_freq}. "
                         "Re-evaluate bound order from live readback if a power-mode switch resets clocks; "
                         "do not continue when a profile rejects the target or requests a reboot.",
                validation="Modelled apply and restore return all four fields to snapshot. Physical switching/rollback not exercised.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",type=int,required=True)
    p.add_argument("--hz",type=int,required=True)
    p.add_argument("--governor",required=True)
    a=p.parse_args()
    write(a.out,proposal(inventory(),a.mode,a.hz,a.governor))
