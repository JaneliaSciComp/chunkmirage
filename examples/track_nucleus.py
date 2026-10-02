"""Follow one nucleus through a two-day time-lapse of a growing stem cell colony, reading only
a box around it from each frame, and write its volume over time: the nuclear growth
trajectories of Dixon et al. 2024 (Allen Institute for Cell Science), from their public
OME-Zarr segmentation, one nucleus at a time and nothing downloaded whole.

    uv run python examples/track_nucleus.py [--frame 100] [--label 148] [--out nucleus.csv]

The segmentation numbers nuclei afresh in every frame, so the nucleus is followed by its
overlap: in each next frame it is the label that covers most of it (``chunkmirage.tracking``,
the same code the browser's track page runs). It is followed both ways from ``--frame``
until it is lost (it leaves the field, or the segmentation misses it). A nucleus dividing
first collapses (mitosis breaks its envelope down) and is lost; its daughters are then
looked for where it was, as new nuclei, and followed on: one row per nucleus and frame,
with its branch and its mother's. Each step reads about twenty planes of a 312 x 456 level
from the bucket, cached as they are read.
"""

from __future__ import annotations

import argparse
import csv
import time

BASE = (
    "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/"
    "hipsc_fov_nuclei_timelapse_data_used_for_analysis/baseline_colonies_fov_timelapse_dataset/"
    "20200323_09_small"
)
SEG = f"{BASE}/seg.ome.zarr"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=SEG, help="an OME-Zarr label image with axes t, c, z, y, x")
    ap.add_argument("--frame", type=int, default=100, help="the frame the nucleus is picked in")
    ap.add_argument("--label", type=int, default=148, help="its label in that frame")
    ap.add_argument("--level", type=int, default=3, help="the level followed (3: 1.08 µm across)")
    ap.add_argument("--margin", default="3,6,6", help="the box read around it, voxels z,y,x")
    ap.add_argument("--out", default="nucleus.csv", help="where the trajectories are written")
    ap.add_argument("--nuclei", type=int, default=8, help="at most this many in the lineage")
    args = ap.parse_args()

    from chunkmirage import open_source, tracking
    from chunkmirage.core import Box

    ms = open_source(args.source, cache_bytes=1 << 30)
    level = ms[args.level]
    info = level.info
    shape, voxel = info.shape[-3:], info.voxel_size[-3:]
    frames = info.shape[0]
    minutes = info.voxel_size[0]  # the t axis: minutes per frame

    def read(t, lo, hi):
        return level.read(Box((t, 0, *lo), (t + 1, 1, *hi)))[0, 0]

    t0 = time.perf_counter()
    margin = [int(m) for m in args.margin.split(",")]
    branches = tracking.lineage(read, shape, voxel, args.frame, args.label, frames, margin, args.nuclei)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nucleus", "mother", "frame", "hours", "label", "volume_um3", "centroid_z", "centroid_y", "centroid_x"])
        for k, b in enumerate(branches):
            mother = b[min(b)].get("mother", "")
            for t in sorted(b):
                r = b[t]
                w.writerow([k, mother, t, round(t * minutes / 60, 3), r["label"], round(r["volume"], 2), *[round(c, 2) for c in r["centroid"]]])
    print(f"followed nucleus {args.label} (frame {args.frame}) and {len(branches) - 1} descendants "
          f"in {time.perf_counter() - t0:.0f} s:")
    for k, b in enumerate(branches):
        lo, hi = min(b), max(b)
        vols = [b[t]["volume"] for t in sorted(b)]
        mother = b[lo].get("mother")
        print(f"  {k}{f' (daughter of {mother})' if mother is not None else ''}: frames {lo}-{hi} "
              f"({(hi - lo) * minutes / 60:.1f} h), {vols[0]:.0f} -> {max(vols):.0f} µm³")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
