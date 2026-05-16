#!/usr/bin/env bash
# Download ROIs1158_spring from SEN12MS-CR
# Grabs: cloudy Sentinel-2 (s2), cloud-free Sentinel-2 (s2_cloudfree), and Sentinel-1 SAR (s1)
# Credentials: m1554803 / m1554803
# Data hosted at: dataserv.ub.tum.de

set -euo pipefail

DEST="${1:-./SEN12MS-CR}"
FTP_USER="m1554803"
FTP_PASS="m1554803"
BASE_URL="ftp://${FTP_USER}:${FTP_PASS}@dataserv.ub.tum.de"

# Archives to download for ROIs1158_spring
# s2       = cloudy Sentinel-2 (13 bands)        — the model input
# s2_cloudfree = cloud-free Sentinel-2 (13 bands) — the ground truth target
# s1       = Sentinel-1 SAR (2 bands)             — auxiliary conditioning input
ARCHIVES=(
    "ROIs1158_spring_s2.tar.gz"
    "ROIs1158_spring_s2_cloudy.tar.gz"
    "ROIs1158_spring_s1.tar.gz"
)

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading ROIs1158_spring to: $(pwd)"
echo "--------------------------------------"

for archive in "${ARCHIVES[@]}"; do
    if [ -f "${archive%.tar.gz}/.done" ]; then
        echo "[skip] ${archive%.tar.gz} already extracted"
        continue
    fi

    echo ""
    echo ">> Downloading $archive ..."
    wget \
        --continue \
        --tries=5 \
        --waitretry=10 \
        --progress=bar:force \
        "${BASE_URL}/${archive}" \
        -O "$archive"

    echo ">> Extracting $archive ..."
    tar -xzf "$archive"

    # Mark as done so re-runs skip extraction
    touch "${archive%.tar.gz}/.done"

    echo ">> Removing archive to free space ..."
    rm "$archive"

    echo ">> Done: $archive"
done

echo ""
echo "======================================="
echo "All ROIs1158_spring files downloaded and extracted."
echo "Location: $(pwd)"
echo "Contents:"
du -sh ROIs1158_spring_* 2>/dev/null || ls -lh