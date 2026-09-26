"""Drive the running Gradio app with Playwright and save UI screenshots to docs/screenshots/.

    python -m footagefind.app &            # in another shell (fresh data/index recommended)
    python scripts/capture_screenshots.py [--url http://127.0.0.1:7860] [--chromium /path/to/chrome]

Indexes data/videos/vtest.avi and indoor_desk.avi through the UI, runs two
searches, clicks a result, and captures the Runtime and Audit tabs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860/")
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        pg.goto(args.url)
        pg.wait_for_timeout(3000)

        def tab(name):
            pg.get_by_role("tab", name=name).click()
            pg.wait_for_timeout(800)

        def shot(name):
            path = OUT / name
            pg.screenshot(path=str(path), full_page=True)
            print("saved", path.relative_to(ROOT))

        if not args.skip_index:
            tab("Index videos")
            for i, video in enumerate(["data/videos/vtest.avi", "data/videos/indoor_desk.avi"]):
                pg.get_by_label("...or pick a file from data/videos").click()
                pg.get_by_role("option", name=video, exact=True).click()
                pg.get_by_role("button", name="Index video").click()
                if i == 0:
                    pg.wait_for_timeout(15000)
                    shot("01_indexing_progress.png")
                pg.locator("strong", has_text="Indexed").filter(has_text=Path(video).name).first.wait_for(timeout=600_000)
                pg.wait_for_timeout(1000)
                if i == 0:
                    shot("02_index_done.png")

        tab("Search")
        box = pg.get_by_placeholder("e.g. person carrying")
        for i, q in enumerate(["a woman in a red jacket", "a man and a woman walking together on the grass"]):
            box.fill(q)
            pg.get_by_role("button", name="Search", exact=True).click()
            pg.locator("text=moments for").first.wait_for(timeout=60_000)
            pg.wait_for_timeout(2500)
            if i == 1:
                pg.locator("button.thumbnail-item").nth(0).click()
                pg.wait_for_timeout(2500)
                pg.evaluate("() => { const v = document.querySelector('#ff-player video'); if (v) v.pause(); }")
            shot(f"0{3 + i}_search_{'red_jacket' if i == 0 else 'couple_on_grass'}.png")

        tab("Runtime")
        pg.get_by_role("button", name="Load all models now").click()
        pg.wait_for_timeout(3000)
        pg.get_by_role("button", name="Probe node placement (ORT profiling)").click()
        pg.locator("text=Nodes by execution provider").wait_for(timeout=120_000)
        pg.wait_for_timeout(1000)
        shot("05_runtime_providers.png")

        tab("Audit log")
        pg.get_by_role("button", name="Refresh").click()
        pg.wait_for_timeout(1500)
        shot("06_audit_log.png")
        browser.close()


if __name__ == "__main__":
    main()
