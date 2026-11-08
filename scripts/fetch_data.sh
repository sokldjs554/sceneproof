#!/usr/bin/env bash
# Downloads the sample/evaluation videos and prepares the transcoded working copies.
# Sources and licences are documented in data/README.md. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/videos/raw"
OUT="$ROOT/data/videos"
mkdir -p "$RAW" "$OUT"

# dl <dest> <url> [<fallback-url>...]: retry each URL, then try the next host. raw.githubusercontent.com serves the
# file directly; the github.com/<repo>/raw/ redirect endpoint returned 504 for 30 s straight in CI (vlm-eval #6/#7).
dl() {
  local dest="$1"; shift
  local u
  for u in "$@"; do
    if curl -fsSL --retry 4 --retry-all-errors --retry-delay 3 -o "$dest.part" "$u"; then mv "$dest.part" "$dest"; return 0; fi
    echo "  [fetch] failed: $u — trying next source" >&2
  done
  rm -f "$dest.part"; echo "[fetch] all sources failed for $(basename "$dest")" >&2; return 22
}
INTEL_REPO="intel-iot-devkit/sample-videos"
INTEL_VIDEOS=(worker-zone-detection people-detection one-by-one-person-detection person-bicycle-car-detection store-aisle-detection)
for v in "${INTEL_VIDEOS[@]}"; do
  if [ ! -s "$RAW/$v.mp4" ]; then
    echo "[fetch] $v.mp4"
    dl "$RAW/$v.mp4" "https://raw.githubusercontent.com/$INTEL_REPO/master/$v.mp4" "https://github.com/$INTEL_REPO/raw/master/$v.mp4"
  fi
done
if [ ! -s "$RAW/vtest.avi" ]; then
  echo "[fetch] vtest.avi (OpenCV samples)"
  dl "$RAW/vtest.avi" "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi" "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi"
fi

# Working copies: H.264 (browser-playable, RTSP passthrough-friendly), constant frame rate,
# capped at 1280x720 so a 4-vCPU box can decode + analyse in real time.
transcode() {
  local src="$1" dst="$2" fps="$3"
  if [ -s "$dst" ]; then return; fi
  echo "[transcode] $(basename "$src") -> $(basename "$dst") @ ${fps}fps"
  ffmpeg -v error -y -i "$src" -vf "scale='min(1280,iw)':-2" -r "$fps" -c:v libx264 -preset veryfast -crf 23 \
    -pix_fmt yuv420p -g "$((fps * 2))" -an -movflags +faststart "$dst"
}
transcode "$RAW/worker-zone-detection.mp4"      "$OUT/worker-zone.mp4"        30
transcode "$RAW/one-by-one-person-detection.mp4" "$OUT/one-by-one.mp4"         10
transcode "$RAW/people-detection.mp4"           "$OUT/people-indoor.mp4"      12
transcode "$RAW/person-bicycle-car-detection.mp4" "$OUT/parking-lot.mp4"      12
transcode "$RAW/store-aisle-detection.mp4"      "$OUT/store-aisle.mp4"        30
transcode "$RAW/vtest.avi"                      "$OUT/campus-walkway.mp4"     10
sha256sum "$OUT"/*.mp4 > "$OUT/SHA256SUMS"
echo "[fetch] done: $(ls "$OUT"/*.mp4 | wc -l) working videos"
