#!/usr/bin/env python3
"""Download an aligned-triplet SEN12MS-CR subset for local dev and testing.

Streams three archives from TUM dataserv FTP (via wget) and extracts a
scene-diverse subset from each, using Python's tarfile so extraction stops
as soon as the quota is met — the multi-GB tarballs never land on disk:

    ROIs1158_spring_s2.tar.gz           cloud-free Sentinel-2  (anchor)
    ROIs1158_spring_s1.tar.gz           Sentinel-1 SAR
    ROIs1158_spring_s2_cloudy.tar.gz    cloudy Sentinel-2

Flow:
  1. Anchor pass — stream s2, extract PER_SCENE patches from each of the
     first N_SCENES distinct scenes, record their patch IDs (<scene>_p<N>).
  2. Filtered passes — stream s1 and s2_cloudy, extract only the .tif whose
     patch ID is in the anchor set.
  3. Reconcile — keep only IDs present in all three subsets, drop orphans.

The patch ID suffix (`_<scene>_p<N>.tif`) is identical across archives, so
it is a stable cross-archive key. End state on disk:

    data/
      ROIs1158_spring_s2/s2_<scene>/...
      ROIs1158_spring_s1/s1_<scene>/...
      ROIs1158_spring_s2_cloudy/s2_cloudy_<scene>/...
      .done

Disk/bandwidth: final footprint is ~1.5 GB (PER_SCENE * N_SCENES * 3 small
patches). Bandwidth is higher than a single-scene grab — reaching
N_SCENES distinct scenes means streaming through the leading ~N_SCENES
scenes of each archive (several GB per archive), since tar is sequential.
Streaming has no resume — a dropped connection restarts that archive.

Usage:
    python download_local_data.py                       # 20 x 10 scenes
    PER_SCENE=10 N_SCENES=5 python download_local_data.py
    rm data/.done && python download_local_data.py      # refresh
"""

import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PER_SCENE = int(os.environ.get("PER_SCENE", "10"))
N_SCENES = int(os.environ.get("N_SCENES", "3"))
DEST = Path("data")
FTP_BASE = "ftp://m1554803:m1554803@dataserv.ub.tum.de"

# (archive filename, extracted top-level directory). s2 (cloud-free) must
# come first — it defines the anchor patch IDs the others are filtered to.
ARCHIVES = [
    ("ROIs1158_spring_s2.tar.gz",        "ROIs1158_spring_s2"),
    ("ROIs1158_spring_s1.tar.gz",        "ROIs1158_spring_s1"),
    ("ROIs1158_spring_s2_cloudy.tar.gz", "ROIs1158_spring_s2_cloudy"),
]

ID_RE = re.compile(r"_(\d+_p\d+)\.tif$")

# tarfile gained a mandatory extraction filter in 3.12; pass it where
# supported, omit it on the project's Python 3.10.
_EXTRACT_KW = {"filter": "data"} if sys.version_info >= (3, 12) else {}


def patch_id(name):
    """Return the <scene>_p<N> id from a patch filename, or None."""
    m = ID_RE.search(name)
    return m.group(1) if m else None


def scene_of(pid):
    """Return the scene part of a patch id ('1_p206' -> '1')."""
    return pid.rsplit("_p", 1)[0]


def scene_quota_picker(per_scene, n_scenes):
    """Stateful picker: accept up to <per_scene> patches from each of the
    first <n_scenes> distinct scenes encountered in the stream.

    Returns (accept, stop) callables for stream_subset. Tar archives are
    laid out scene-by-scene, so once an (n_scenes+1)-th scene appears the
    chosen scenes are all behind us and we can stop.
    """
    counts = {}
    overflow = False

    def accept(pid):
        nonlocal overflow
        scene = scene_of(pid)
        if scene in counts:
            if counts[scene] < per_scene:
                counts[scene] += 1
                return True
            return False
        if len(counts) >= n_scenes:
            overflow = True
            return False
        counts[scene] = 1
        return True

    def stop(_extracted):
        if overflow:
            return True
        return (len(counts) == n_scenes
                and all(c >= per_scene for c in counts.values()))

    return accept, stop


def stream_subset(archive, accept, stop):
    """Stream <archive> from FTP via wget and extract members for which
    accept(patch_id) is True, stopping once stop(extracted_ids) is True.

    Returns the set of extracted patch IDs.
    """
    url = f"{FTP_BASE}/{archive}"
    proc = subprocess.Popen(
        ["wget", "-q", "-O", "-", "--tries=3", url],
        stdout=subprocess.PIPE,
    )
    extracted = set()
    try:
        # r|gz = streaming (non-seekable) mode, correct for a pipe.
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".tif"):
                    continue
                pid = patch_id(member.name)
                if pid is None or pid in extracted:
                    continue
                if accept(pid):
                    tar.extract(member, path=DEST, **_EXTRACT_KW)
                    extracted.add(pid)
                # Checked on every member (extract or skip) so the picker's
                # overflow signal is caught even when nothing was extracted.
                if stop(extracted):
                    break
    finally:
        # Sever the stream: closing the read end makes wget see SIGPIPE.
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return extracted


def ids_in(subdir):
    """Set of patch IDs currently extracted under DEST/<subdir>."""
    return {
        pid
        for tif in (DEST / subdir).rglob("*.tif")
        if (pid := patch_id(tif.name)) is not None
    }


def main():
    if (DEST / ".done").exists():
        print(f"[skip] {DEST}/ already populated (rm {DEST}/.done to refresh)")
        return

    # Clear any partial state from a previous (possibly broken) run.
    for _, subdir in ARCHIVES:
        shutil.rmtree(DEST / subdir, ignore_errors=True)
    DEST.mkdir(exist_ok=True)

    # Anchor pass: PER_SCENE patches from each of N_SCENES scenes of s2.
    s2_archive, _ = ARCHIVES[0]
    print(f"==> {s2_archive} (anchor: {PER_SCENE} patches x {N_SCENES} scenes)")
    accept, stop = scene_quota_picker(PER_SCENE, N_SCENES)
    anchor = stream_subset(s2_archive, accept, stop)
    n_scenes_got = len({scene_of(p) for p in anchor})
    print(f"    anchored {len(anchor)} patch IDs across {n_scenes_got} scenes")
    if not anchor:
        sys.exit(f"ERROR: no .tif extracted from {s2_archive} "
                 f"(wget/FTP failure?)")

    # Filtered passes: s1 and s2_cloudy, restricted to the anchor IDs.
    anchor_n = len(anchor)
    for archive, subdir in ARCHIVES[1:]:
        print(f"==> {archive} (filtered to anchor IDs)")
        got = stream_subset(archive,
                            accept=lambda pid: pid in anchor,
                            stop=lambda e: len(e) >= anchor_n)
        print(f"    extracted {len(got)}")

    # Reconcile strict triplets: keep only IDs present in all three.
    common = anchor & ids_in(ARCHIVES[1][1]) & ids_in(ARCHIVES[2][1])
    print(f"==> reconciling — {len(common)} aligned triplets")
    for _, subdir in ARCHIVES:
        for tif in (DEST / subdir).rglob("*.tif"):
            if patch_id(tif.name) not in common:
                tif.unlink()

    (DEST / ".done").touch()
    print(f"\n=======================================")
    print(f"Local subset ready at: {DEST.resolve()}")
    print(f"Aligned triplets: {len(common)} "
          f"across {len({scene_of(p) for p in common})} scenes")


if __name__ == "__main__":
    main()
