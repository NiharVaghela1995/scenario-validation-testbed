"""Download NGSIM vehicle trajectories from the US DOT open-data portal (public data) into data/raw/ngsim/.

    python pilot/download_ngsim.py --location us-101                         # one site, about 4.8 million rows
    python pilot/download_ngsim.py --location us-101 --minutes 1             # first minute only (smoke test)

The file is written in pages and resumes after an interruption. Nothing downloaded is ever committed (data/raw/ is
git-ignored). Only the freeway sites us-101 and i-80 are in scope.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data.transportation.gov/resource/8ect-6jqj"
PAGE = 50_000


def get(url, retries=4):
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "scenario-validation-testbed"}), timeout=180) as r:
                return r.read().decode("utf-8")
        except OSError:
            if k == retries:
                raise
            time.sleep(5 * 2 ** k)


def query(fmt, **params):
    return f"{BASE}.{fmt}?" + urllib.parse.urlencode({f"${k}": v for k, v in params.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--location", choices=("us-101", "i-80"), required=True)
    ap.add_argument("--minutes", type=float, help="only this many minutes from the start of the first study period")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "ngsim"))
    a = ap.parse_args(argv)
    where = f"location='{a.location}'"
    name = a.location
    if a.minutes:
        rows = json.loads(get(query("json", select="min(global_time) AS t0", where=where)))
        t0 = int(float(rows[0]["t0"]))
        where += f" AND global_time >= {t0} AND global_time < {t0 + int(a.minutes * 60000)}"
        name += f"_{a.minutes:g}min"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    done = max(sum(1 for _ in path.open()) - 1, 0) if path.exists() else 0
    with path.open("a") as f:
        while True:
            text = get(query("csv", where=where, order="vehicle_id,global_time", limit=PAGE, offset=done))
            head, _, body = text.partition("\n")
            if done == 0:
                f.write(head + "\n")
            n = body.count("\n")
            if n == 0:
                break
            f.write(body)
            f.flush()
            done += n
            print(f"{done:,} rows", flush=True)
            if n < PAGE:
                break
    print("written", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name)


if __name__ == "__main__":
    main()
