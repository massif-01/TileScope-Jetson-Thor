"""Metadata-only validation of all safetensors shards referenced by the Thor model index."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

from tilescope.common import write_json
from tilescope.environment import DEFAULT_MODEL

ELEMENT_BYTES={"F32":4,"BF16":2,"U8":1,"F8_E4M3":1}


def inspect(root):
    index=json.loads((root/"model.safetensors.index.json").read_bytes())
    mapping=index["weight_map"]
    if not mapping:raise ValueError("empty weight map")
    shards={}
    seen=set()
    for name in sorted(set(mapping.values())):
        if Path(name).name!=name:raise ValueError("invalid shard name")
        path=root/name
        with path.open("rb") as f:
            size,=struct.unpack("<Q",f.read(8))
            if not (2<=size<=100_000_000):raise ValueError("invalid header size")
            header=f.read(size)
        entries=json.loads(header); entries.pop("__metadata__",None)
        expected={k for k,v in mapping.items() if v==name}
        if seen.intersection(entries) or expected!=set(entries):raise ValueError("index/shard tensor names disagree")
        seen.update(entries); kinds=Counter();sizes=Counter();offset=0
        for tensor,v in sorted(entries.items(),key=lambda item:item[1]["data_offsets"]):
            dtype=v["dtype"];shape=v["shape"];start,end=v["data_offsets"]
            if (not all(type(n) is int and n>=0 for n in shape) or start!=offset or
                end-start!=math.prod(shape)*ELEMENT_BYTES[dtype]):raise ValueError(f"non-contiguous or invalid tensor {tensor}")
            offset=end;kinds[dtype]+=1;sizes[dtype]+=end-start
        if 8+size+offset!=path.stat().st_size:raise ValueError("shard file length mismatch")
        shards[name]=dict(header_sha256=hashlib.sha256(header).hexdigest(),header_bytes=size,payload_bytes=offset,
                          dtype_counts=dict(kinds),payload_bytes_by_dtype=dict(sizes),tensor_count=len(entries))
    return dict(schema="tilescope.checkpoint.v1",scope="index, shard headers and byte ranges",
                tensors=len(seen),shards=shards,index_metadata=index.get("metadata",{}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(DEFAULT_MODEL));p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();write_json(args.out,inspect(args.root))
