#!/usr/bin/env python3
"""Print docs/paper.html to docs/courtsight-paper.pdf with headless Chromium (pip install playwright).

Run after build_site.py. Set CHROMIUM to a browser binary if Playwright's own is not installed.
"""
from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> None:
    exe = os.environ.get("CHROMIUM")
    with sync_playwright() as p:
        b = p.chromium.launch(**({"executable_path": exe} if exe else {}))
        pg = b.new_page(viewport={"width": 760, "height": 1000}, color_scheme="light")
        pg.goto((HERE / "docs" / "paper.html").as_uri())
        pg.emulate_media(media="print")
        pg.wait_for_timeout(400)
        pg.pdf(path=str(HERE / "docs" / "courtsight-paper.pdf"), format="A4", print_background=True,
               margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#888">'
                               'CourtSight working paper · <span class="pageNumber"></span>/<span class="totalPages"></span></div>')
        b.close()
    print("wrote docs/courtsight-paper.pdf")


if __name__ == "__main__":
    main()
