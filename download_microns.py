"""Download the MICrONS functional-connectomics node/edge tables (~500 MB).

Source: public BossDB bucket, release accompanying Ding et al. (2025),
"Functional connectomics reveals general wiring rule in mouse visual cortex",
Nature, doi:10.1038/s41586-025-08840-3.

    s3://bossdb-open-data/iarpa_microns/minnie/functional_data/
        functional_connectomics/node_and_edge_properties/v1/

No AWS credentials are needed; files are fetched over HTTPS.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/"
        "functional_connectomics/node_and_edge_properties/v1/")
FILES = {  # name: expected size in bytes (checked 2026-11)
    "node_data_v1.pkl": 294_635_032,
    "edge_data_v1.pkl": 201_511_086,
    "readme_v1.md": None,
}


def download(dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        target = dest / name
        if target.exists() and not force and (size is None or target.stat().st_size == size):
            print(f"ok   {target} (cached)")
            continue
        print(f"get  {BASE + name}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(BASE + name) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        if size is not None and tmp.stat().st_size != size:
            sys.exit(f"size mismatch for {name}: got {tmp.stat().st_size}, expected {size}")
        tmp.replace(target)
        print(f"ok   {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--force", action="store_true")
    download(**vars(ap.parse_args()))
