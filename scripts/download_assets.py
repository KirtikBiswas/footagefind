"""Download model weights and test videos (GitHub-hosted; works where Hugging Face is blocked).

    python scripts/download_assets.py            # everything
    python scripts/download_assets.py --videos   # test clips only (enough on a device that gets
                                                 # pre-exported ONNX models copied over)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = [
    ("https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
     "vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
     ROOT / "models" / "vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
     "46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa"),
    ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
     ROOT / "models" / "yolov8n.pt",
     "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
]
VIDEOS = [
    ("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
     ROOT / "data" / "videos" / "vtest.avi",
     "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"),
    ("https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/video/1920x1080.avi",
     ROOT / "data" / "videos" / "indoor_desk.avi",
     "42f77d89c19428a06c70e3c8f455cc3cced40bd0b46dafee3bfbfde323362153"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, digest: str) -> None:
    if dst.exists() and sha256(dst) == digest:
        print("ok      ", dst.relative_to(ROOT))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("download", url)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {dst.name}: {got}")
    tmp.replace(dst)
    print("saved   ", dst.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true", help="only the test videos")
    args = ap.parse_args()
    for url, dst, digest in (VIDEOS if args.videos else WEIGHTS + VIDEOS):
        fetch(url, dst, digest)


if __name__ == "__main__":
    main()
