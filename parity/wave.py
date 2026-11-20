"""Fresh client process for an explicitly paired service comparison."""
import json
from pathlib import Path
import sys
from .service import run_wave
from tilescope.common import read_json

if __name__=="__main__":
    root=Path(sys.argv[1])
    results=run_wave(json.loads(sys.argv[2]),read_json(root/"plan.json"),root,read_json(root/"inputs.json"))
    sys.exit(0 if all(r["status"]=="passed" for r in results) else 1)
