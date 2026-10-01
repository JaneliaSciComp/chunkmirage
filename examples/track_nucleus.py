"""Follow one nucleus through a two-day time-lapse of a growing stem cell colony, reading only
a box around it from each frame, and write its volume over time: the nuclear growth
trajectories of Dixon et al. 2024 (Allen Institute for Cell Science), from their public
OME-Zarr segmentation, one nucleus at a time and nothing downloaded whole.

    uv run python examples/track_nucleus.py [--frame 100] [--label 148] [--out nucleus.csv]

The segmentation numbers nuclei afresh in every frame, so the nucleus is followed by its
overlap: in each next frame it is the label that covers most of it (``chunkmirage.tracking``,
the same code the browser's track page runs). It is followed both ways from ``--frame``
until it is lost (it leaves the field, or the segmentation misses it); when it divides, one
daughter is followed on and the row says so. Each step reads about twenty planes of a
312 x 456 level from the bucket, cached as they are read.
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
    ap.add_argument("--out", default="nucleus.csv", help="where the trajectory is written")
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
    rows = {}
    for frames_ in (range(args.frame + 1, frames), range(args.frame - 1, -1, -1)):
        for t, r in tracking.follow(read, shape, voxel, args.frame, args.label, frames_, [int(m) for m in args.margin.split(",")]):
            rows[t] = r
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "hours", "label", "volume_um3", "centroid_z", "centroid_y", "centroid_x", "divided"])
        for t in sorted(rows):
            r = rows[t]
            w.writerow([t, round(t * minutes / 60, 3), r["label"], round(r["volume"], 2), *[round(c, 2) for c in r["centroid"]], r.get("divided", False)])
    span = (min(rows), max(rows))
    vols = [rows[t]["volume"] for t in sorted(rows)]
    divisions = [t for t in sorted(rows) if rows[t].get("divided")]
    print(f"followed nucleus {args.label} (frame {args.frame}) through frames {span[0]}-{span[1]} "
          f"({(span[1] - span[0]) * minutes / 60:.1f} h) in {time.perf_counter() - t0:.0f} s")
    print(f"volume {vols[0]:.0f} -> {vols[-1]:.0f} µm³ (largest {max(vols):.0f})"
          + (f"; divided at frame{'s' if len(divisions) > 1 else ''} {', '.join(map(str, divisions))}" if divisions else ""))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
