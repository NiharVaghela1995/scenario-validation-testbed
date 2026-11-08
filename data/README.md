# Data

**No third-party data or raw extracts are stored in this repository.** `data/raw/` is git-ignored.

## Synthetic data (included)

`synthetic/cutin_events_g1.csv` and `synthetic/cutin_events_g2.csv` are 400 synthetic cut-ins per generator, written by
`experiments/generation_cutin.py` from the generators in `critical_events/cutin_chain.py`. The short synthetic driving logs are
produced on demand by `critical_events/synthetic.py`.

## NGSIM (download script)

Public US Department of Transportation data. The script pages through the open-data API and writes to `data/raw/ngsim/`:

```bash
python pilot/download_ngsim.py --location us-101      # about 4.8 million rows
python pilot/download_ngsim.py --location us-101 --minutes 1   # first minute, for a quick trial
```

Dataset page: <https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj>.
The raw trajectories have well-documented errors in speed and position (Montanino & Punzo 2013; Coifman & Li 2017). The loader does
not repair them.

## highD (download yourself under the dataset's licence)

Access is by application and for non-commercial use only: <https://www.highd-dataset.com/>. You may not redistribute the data or
anything from which it can be recovered, and you must cite Krajewski et al., ITSC 2018 (doi:10.1109/ITSC.2018.8569552). Read the
full terms on the dataset page. Once you have access, put the files in `data/raw/highd/` and use `pilot/loaders.py:load_highd`:

```text
data/raw/highd/01_tracks.csv  01_tracksMeta.csv  01_recordingMeta.csv  ...
```

## Argoverse 2 Sensor (download yourself under the dataset's licence)

Licensed CC BY-NC-SA 4.0: <https://www.argoverse.org/av2.html>. Download logs yourself under those terms and use
`pilot/loaders.py:load_av2`, which needs `annotations.feather` and `city_SE3_egovehicle.feather` of a log. The dataset guide shows how to
fetch the data with `s5cmd`: <https://argoverse.github.io/user-guide/getting_started.html>.

```text
data/raw/av2/<log_id>/annotations.feather  city_SE3_egovehicle.feather  ...
```

Before publishing anything derived from these datasets, check the dataset's terms.
