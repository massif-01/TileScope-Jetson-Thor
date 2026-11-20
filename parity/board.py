"""Identify physical RAM and BPMP clocks using the device's existing read-only sudo access."""
import argparse
from pathlib import Path
import subprocess
import time
from tilescope.common import device_slot, read_json, write_json

BASE="/sys/kernel/debug/bpmp/debug/clk"
CLOCKS=("emc","vic","pva0_vps","pva0_cpu_axi","isp","isp1","nvenc","nvenc0","nvenc1","nvdec","nvdec0","nvdec1","ofa","nvjpg","nvjpg1")
MEMORY_TYPES={0x18:"DDR3",0x1a:"DDR4",0x1b:"LPDDR",0x1c:"LPDDR2",0x1d:"LPDDR3",
              0x1e:"LPDDR4",0x22:"DDR5",0x23:"LPDDR5"}
TABLE="/sys/firmware/dmi/tables/DMI"


def read_as_allowed(argv):
    completed=subprocess.run(["sudo","-n",*argv],capture_output=True,timeout=10)
    return completed.returncode,completed.stdout,completed.stderr.decode(errors="replace")


def physical_ram():
    rc,data,err=read_as_allowed(["cat",TABLE])
    if rc:return {"source":TABLE,"status":"query_failed","reason":err}
    offset=0;devices=[];types=[]
    while offset+4<=len(data):
        kind,length=data[offset:offset+2]
        if length<4 or offset+length>len(data):raise ValueError("invalid SMBIOS structure length")
        types.append(kind);block=data[offset:offset+length]
        if kind==17:
            if length<0x13:raise ValueError("short SMBIOS memory device record")
            devices.append({"memory_type_code":block[0x12],"memory_type":MEMORY_TYPES.get(block[0x12],"unspecified or other"),
                            "total_width_raw_bits":int.from_bytes(block[8:10],"little"),
                            "data_width_raw_bits":int.from_bytes(block[10:12],"little"),
                            "smbios_size_field_raw":int.from_bytes(block[12:14],"little")})
        end=data.find(b"\0\0",offset+length)
        if end<0:raise ValueError("unterminated SMBIOS string table")
        offset=end+2
        if kind==127:break
    return {"source":TABLE,"status":"queried","read_bytes":len(data),"record_types":types,"type17":devices,
            "limits":"SMBIOS Type 17 may describe LPDDR5 without disambiguating LPDDR5X. Width 0xFFFF means unknown. SKU and theoretical bandwidth are separate."}


def capture(args):
    if args.samples<1 or args.interval<=0:raise ValueError("positive sample count and interval required")
    with device_slot():
        path=args.out.resolve();path.mkdir(parents=True,exist_ok=False)
        rc,data,err=read_as_allowed(["ls","-1",BASE]);available=set(data.decode().splitlines())
        selection={k:("available" if k in available else "not_listed" if rc==0 else "query_failed") for k in CLOCKS}
        proof={"schema":"tilescope.board.read.v1","directory":BASE,"directory_rc":rc,
               "directory_error":err,"directory_entries":sorted(available),"clock_objects":selection,"clocks_unit":"Hz",
               "method":"sudo -n ls/cat on fixed sysfs or debugfs paths; no device writes","physical_ram":physical_ram(),"samples":[]}
        keys=[k for k in CLOCKS if k in available]
        for i in range(args.samples):
            start=time.time_ns()
            if keys:
                result,values,err=read_as_allowed(["cat",*[f"{BASE}/{k}/rate" for k in keys]])
            else:result,values,err=rc,b"",err
            vals=values.decode().splitlines()
            exact=result==0 and len(vals)==len(keys) and all(v.isdigit() for v in vals)
            proof["samples"].append({"index":i,"begin_ns":start,"end_ns":time.time_ns(),"read_rc":result,
                "stderr":err,"raw":vals,"rates_hz":{k:int(v) for k,v in zip(keys,vals)} if exact else None})
            if i+1<args.samples:time.sleep(args.interval)
        write_json(path/"board.json",proof)
    return int(rc!=0 or not keys or not all(s["rates_hz"] for s in proof["samples"]))


def review(path):
    p=read_json(Path(path)/"board.json")
    if p["schema"]!="tilescope.board.read.v1" or p["directory_rc"]!=0:raise ValueError("BPMP directory query failed")
    expected=[k for k in CLOCKS if k in p["directory_entries"]]
    if not expected or p["physical_ram"]["status"]!="queried" or not p["samples"]:raise ValueError("physical inventory incomplete")
    for i,row in enumerate(p["samples"]):
        if row["index"]!=i or row["end_ns"]<row["begin_ns"] or row["read_rc"]!=0:raise ValueError("invalid clock sample")
        if (len(row["raw"])!=len(expected) or {k:int(v) for k,v in zip(expected,row["raw"])}!=row["rates_hz"]):
            raise ValueError("BPMP clock value/label mismatch")
    return p

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--samples",type=int,default=4);p.add_argument("--interval",type=float,default=.5)
    raise SystemExit(capture(p.parse_args()))
