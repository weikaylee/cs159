#!/usr/bin/env python3
"""Download and extract the full spring SEN12MS-CR triplet.

Streams three archives from TUM dataserv FTP (via wget) and extracts every
`.tif` member from each, using Python's tarfile so the multi-GB tarballs
never land on disk:

    ROIs1158_spring_s2.tar.gz           cloud-free Sentinel-2  (anchor)
    ROIs1158_spring_s1.tar.gz           Sentinel-1 SAR
    ROIs1158_spring_s2_cloudy.tar.gz    cloudy Sentinel-2

End state on disk:

    data/
      ROIs1158_spring_s2/...
      ROIs1158_spring_s1/...
      ROIs1158_spring_s2_cloudy/...
      .done

Disk/bandwidth: final footprint is the full dataset size for these three
archives. Streaming has no resume — a dropped connection restarts that
archive.

Usage:
    python download_all_data.py
    rm data/.done && python download_all_data.py      # refresh
"""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEST = Path("data")
FTP_BASE = "ftp://m1554803:m1554803@dataserv.ub.tum.de"

# (archive filename, extracted top-level directory).
ARCHIVES = [
    ("ROIs1158_spring_s2.tar.gz",        "ROIs1158_spring_s2"),
    ("ROIs1868_summer_s2.tar.gz",        "ROIs1868_summer_s2"),
    ("ROIs1970_fall_s2.tar.gz",          "ROIs1970_fall_s2"),
    ("ROIs2017_winter_s2.tar.gz",        "ROIs2017_winter_s2"),

    ("ROIs1158_spring_s2_cloudy.tar.gz", "ROIs1158_spring_s2_cloudy"),
    ("ROIs1868_summer_s2_cloudy.tar.gz", "ROIs1868_summer_s2_cloudy"),
    ("ROIs1970_fall_s2_cloudy.tar.gz",   "ROIs1970_fall_s2_cloudy"),
    ("ROIs2017_winter_s2_cloudy.tar.gz", "ROIs2017_winter_s2_cloudy"),

    ("ROIs1158_spring_s1.tar.gz",        "ROIs1158_spring_s1"),
    ("ROIs1868_summer_s1.tar.gz",        "ROIs1868_summer_s1"),
    ("ROIs1970_fall_s1.tar.gz",          "ROIs1970_fall_s1"),
    ("ROIs2017_winter_s1.tar.gz",        "ROIs2017_winter_s1"),
]

# tarfile gained a mandatory extraction filter in 3.12; pass it where
# supported, omit it on the project's Python 3.10.
_EXTRACT_KW = {"filter": "data"} if sys.version_info >= (3, 12) else {}


def stream_archive(archive):
    """Stream <archive> from FTP via wget and extract every .tif member."""
    url = f"{FTP_BASE}/{archive}"
    proc = subprocess.Popen(
        ["wget", "-q", "-O", "-", "--tries=3", url],
        stdout=subprocess.PIPE,
    )
    try:
        # r|gz = streaming (non-seekable) mode, correct for a pipe.
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".tif"):
                    continue
                tar.extract(member, path=DEST, **_EXTRACT_KW)
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


def main():
    if (DEST / ".done").exists():
        print(f"[skip] {DEST}/ already populated (rm {DEST}/.done to refresh)")
        return

    # Clear any partial state from a previous (possibly broken) run.
    for archive, subdir in ARCHIVES:
        shutil.rmtree(DEST / subdir, ignore_errors=True)
    DEST.mkdir(exist_ok=True)

    for archive, subdir in ARCHIVES:
        print(f"==> {archive} (full extraction)")
        stream_archive(archive)
        print(f"    extracted into {DEST / subdir}")

    (DEST / ".done").touch()
    print(f"\n=======================================")
    print(f"Local subset ready at: {DEST.resolve()}")
    print("Full spring triplet extracted")


if __name__ == "__main__":
    main()
