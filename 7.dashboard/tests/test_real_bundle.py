"""Opt-in verification of the real pilot bundle, never synthetic fixture images.

Set PCCMA_DASHBOARD_BUNDLE to a complete build directory. These tests read the
original warehouse hash list from the local pilot cache, but never require SSH
or modify the bundle. Optional screenshots/evidence go to PCCMA_DASHBOARD_EVIDENCE.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urljoin

import pytest
from playwright.sync_api import expect

from test_browser import browser, origin  # noqa: F401 (shared pytest fixtures)


@pytest.fixture(scope="module")
def real_pilot():
    configured = os.environ.get("PCCMA_DASHBOARD_BUNDLE")
    if not configured:
        pytest.skip("Set PCCMA_DASHBOARD_BUNDLE to opt into real warehouse-image verification")
    bundle = Path(configured).expanduser().resolve()
    html = (bundle / "index.html").read_text()
    payload = json.loads(html.split('<script type="application/json" id="report-data">', 1)[1].split("</script>", 1)[0])
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert payload["images"]["status"] == "bundled", "A diagnostic preview is not a real crop bundle"
    assert payload["plate"] == "BR00150695"
    assert payload["run_id"] == "BR00150695-20261113T155150Z-g790ecb4"
    assert payload["summary"]["umap_cells"] == len(payload["points"]) == 6500
    assert payload["images"]["files"] == len(manifest["assets"]) == 39000
    return bundle, payload, manifest


def test_real_pilot_all_assets_match_original_warehouse_hashes(real_pilot):
    bundle, payload, manifest = real_pilot
    root = Path(__file__).resolve().parents[2]
    configured_hashes = os.environ.get("PCCMA_DASHBOARD_SOURCE_HASHES")
    hashes = (Path(configured_hashes).expanduser().resolve() if configured_hashes else
              root / "7.dashboard/previews" / (payload["run_id"] + "-warehouse") / "jpg-sha256.txt.gz")
    assert hashes.is_file(), "Real source hashes are required, not only self-generated bundle hashes"
    assets = {asset["path"]: asset for asset in manifest["assets"]}
    referenced = {unquote(path) for point in payload["points"] for path in point["images"].values()}
    assert len(assets) == len(referenced) == 39000
    assert set(assets) == referenced
    original = {}
    with gzip.open(hashes, "rt") as stream:
        for line in stream:
            digest, relative = line.rstrip("\n").split("  ", 1)
            if "assets/" + relative in referenced:
                assert relative not in original
                original[relative] = digest
    assert len(original) == len(assets)
    assert hashlib.sha256((bundle / "index.html").read_bytes()).hexdigest() == manifest["index_sha256"]
    for relative, asset in assets.items():
        path = bundle / relative
        assert not path.is_symlink() and path.is_file()
        assert path.stat().st_size == asset["bytes"]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == asset["sha256"] == original[relative.removeprefix("assets/")]


def isolated_target(page, line):
    """Choose a well-separated real point so pointer rounding cannot hit a neighbor."""
    return page.evaluate("""line => {
        const points=dashboard.visiblePoints;
        const projected=points.map(point=>({point,...dashboard.projection.forward(point)}));
        let best=null, bestDistance=-1;
        for(const candidate of projected.filter(item=>item.point.line===line)){
            let minimum=Infinity;
            for(const other of projected){
                if(other.point.id===candidate.point.id) continue;
                minimum=Math.min(minimum,(candidate.x-other.x)**2+(candidate.y-other.y)**2);
            }
            if(minimum>bestDistance){best=candidate;bestDistance=minimum;}
        }
        const rect=document.getElementById('plot').getBoundingClientRect();
        return {id:best.point.id,x:best.x+rect.left,y:best.y+rect.top};
    }""", line)


def test_real_pilot_native_hover_all_lines_channels_mobile_and_file_open(real_pilot, browser, origin, tmp_path):
    bundle, payload, _manifest = real_pilot
    link = tmp_path / "pccma" / "report"
    link.parent.mkdir()
    link.symlink_to(bundle, target_is_directory=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    errors, failed, requested = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("response", lambda response: failed.append(response.url) if response.status >= 400 else None)
    page.on("request", lambda request: requested.append(request.url))
    page.goto(origin + "/pccma/report/", wait_until="load")
    expect(page.locator("#error-banner")).to_be_hidden()
    expect(page.locator("#image-warning")).to_be_hidden()
    points_by_id = {point["id"]: point for point in payload["points"]}
    representatives = sorted({(point["group"], point["line"]) for point in payload["points"]})
    checked = []
    for group, line in representatives:
        page.locator("#group-select").select_option(group)
        for channel in ["DNA", "ER", "RNA", "AGP", "Mito"]:
            box = page.locator(f"#channel-toggle-{channel}")
            if box.is_enabled() and not box.is_checked():
                box.check()
        target = isolated_target(page, line)
        page.mouse.move(target["x"], target["y"])
        expect(page.locator("#cell-id")).to_have_text(target["id"])
        expect(page.locator("#inspector-title")).to_have_text(line)
        expect(page.locator("#crop-area img")).to_have_count(5)
        assert page.locator("#crop-area img").evaluate_all("images=>images.every(img=>img.complete&&img.naturalWidth===224&&img.naturalHeight===224)")
        assert page.locator(".image-error").count() == 0
        sources = page.locator("#crop-area img").evaluate_all("images=>images.map(img=>img.src)")
        assert set(sources) == {urljoin(origin + "/pccma/report/", path) for channel, path in points_by_id[target["id"]]["images"].items() if channel != "composite"}
        page.mouse.click(target["x"], target["y"])
        expect(page.locator("#pin-status")).to_have_text("Pinned")
        for channel in ["DNA", "ER", "RNA", "AGP", "Mito"]:
            for other in ["DNA", "ER", "RNA", "AGP", "Mito"]:
                box = page.locator(f"#channel-toggle-{other}")
                if other == channel and not box.is_checked():
                    box.check()
                elif other != channel and box.is_checked():
                    box.uncheck()
            # A single channel is still a live canvas mix (tinted), not the five-channel composite.
            expect(page.locator("#crop-area canvas")).to_have_count(1)
            expect(page.locator("#crop-area img")).to_have_count(1)
            expected = points_by_id[target["id"]]["images"]
            assert page.locator("#crop-area img").evaluate_all("images=>images.map(img=>img.src)") == [
                urljoin(origin + "/pccma/report/", expected[channel]),
            ]
        for channel in ["DNA", "ER", "RNA", "AGP", "Mito"]:
            box = page.locator(f"#channel-toggle-{channel}")
            if not box.is_checked():
                box.check()
        expect(page.locator("#crop-area canvas")).to_have_count(1)
        expect(page.locator("#crop-area img")).to_have_count(5)
        checked.append({"group": group, "line": line, "cell_id": target["id"], "channels": 6})
    assert len(checked) == 13
    assert len({item["group"] for item in checked}) == 7
    assert not errors and not failed
    assert all(url.startswith(origin) for url in requested)
    evidence = os.environ.get("PCCMA_DASHBOARD_EVIDENCE")
    if evidence:
        directory = Path(evidence)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / "desktop.png"), full_page=True)
    # Touch selection with real, decoded images at mobile width.
    context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    mobile = context.new_page()
    mobile.goto(origin + "/pccma/report/", wait_until="load")
    first_line = mobile.evaluate("dashboard.visiblePoints[0].line")
    mobile.locator("#plot").scroll_into_view_if_needed()
    target = isolated_target(mobile, first_line)
    assert mobile.evaluate("p => p.x > 0 && p.x < innerWidth && p.y > 0 && p.y < innerHeight", target)
    mobile.touchscreen.tap(target["x"], target["y"])
    expect(mobile.locator("#pin-status")).to_have_text("Pinned")
    expect(mobile.locator("#crop-area img")).to_have_count(5)
    assert mobile.evaluate("document.documentElement.scrollWidth<=innerWidth")
    if evidence:
        mobile.screenshot(path=str(Path(evidence) / "mobile.png"), full_page=True)
    context.close()
    direct = browser.new_page()
    direct.goto((bundle / "index.html").as_uri(), wait_until="load")
    direct.locator("#next-cell").click()
    expect(direct.locator("#crop-area img")).to_have_count(5)
    assert direct.locator("#crop-area img").evaluate_all("images=>images.every(img=>img.naturalWidth===224)")
    direct.close()
    if evidence:
        (Path(evidence) / "browser-verification.json").write_text(json.dumps({
            "plate": payload["plate"], "run_id": payload["run_id"], "status": "passed",
            "checked_real_cells": checked, "groups_checked": 7, "lines_checked": len(checked),
            "channels_per_cell": 6, "native_mouse_hover": True, "click_to_pin": True,
            "channel_switches_checked": ["DNA", "ER", "RNA", "AGP", "Mito"],
            "mobile_touch": True, "direct_file_open": True, "project_prefix": "/pccma/report/",
            "page_errors": errors, "http_errors": failed,
        }, indent=2) + "\n")
    page.close()


def test_real_pilot_hover_crop_is_visible_in_laptop_window(real_pilot, browser, origin, tmp_path):
    bundle, _payload, _manifest = real_pilot
    link = tmp_path / "pccma" / "hover"
    link.parent.mkdir()
    link.symlink_to(bundle, target_is_directory=True)
    page = browser.new_page(viewport={"width": 1280, "height": 577})
    page.goto(origin + "/pccma/hover/", wait_until="load")
    targets = page.evaluate("""() => {
        const rect=document.getElementById('plot').getBoundingClientRect();
        const found=[];
        for(const point of dashboard.visiblePoints){
            const at=dashboard.projection.forward(point);
            const x=at.x+rect.left, y=at.y+rect.top;
            if(x>rect.left+20 && x<rect.right-20 && y>rect.top+20 && y<innerHeight-20)
                found.push({x,y,id:point.id});
            if(found.length===2) break;
        }
        return found;
    }""")
    assert len(targets) == 2, "At least two real plotted cells must be in the initial viewport"
    target, other = targets
    page.mouse.move(target["x"], target["y"])
    preview = page.locator("#hover-preview")
    expect(preview).to_be_visible()
    expect(preview.locator("canvas")).to_have_count(1)
    assert preview.locator("canvas").evaluate("""image => {
        const rect=image.getBoundingClientRect();
        return image.width===224 && rect.width>=100 && rect.height>=100 &&
            rect.top>=0 && rect.left>=0 && rect.bottom<=innerHeight && rect.right<=innerWidth;
    }""")
    expected = page.evaluate("dashboard.hoveredPoint.id")
    assert expected in preview.locator("canvas").get_attribute("aria-label")
    assert preview.evaluate("node=>getComputedStyle(node).pointerEvents") == "none"
    evidence = os.environ.get("PCCMA_DASHBOARD_EVIDENCE")
    if evidence:
        Path(evidence).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(evidence) / "hover-laptop.png"))
    page.mouse.click(target["x"], target["y"])
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    expect(page.locator("#cell-id")).to_have_text(target["id"])
    # Hovering the pinned cell itself shows nothing new; the floating preview stays hidden.
    expect(preview).to_be_hidden()
    # Hovering a *different* cell while pinned still floats it for comparison, and the
    # pinned inspector below is undisturbed.
    page.mouse.move(other["x"], other["y"])
    expect(preview).to_be_visible()
    expected_other = page.evaluate("dashboard.hoveredPoint.id")
    expect(preview.locator("canvas")).to_have_count(1)
    assert expected_other in preview.locator("canvas").get_attribute("aria-label")
    assert expected_other != expected
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    page.close()


@pytest.mark.parametrize("prefix", ["", "/pccma/report"])
def test_real_pilot_mixes_grayscale_for_hover_and_pin_and_serves_catalog(
        real_pilot, browser, origin, tmp_path, prefix):
    from test_channel_mix import expected_pixel, wait_mix

    bundle, payload, manifest = real_pilot
    if prefix:
        link = tmp_path / prefix.lstrip("/")
        link.parent.mkdir(parents=True)
        link.symlink_to(bundle, target_is_directory=True)
    else:
        for path in bundle.iterdir():
            (tmp_path / path.name).symlink_to(path, target_is_directory=path.is_dir())
    page = browser.new_page(viewport={"width": 1600, "height": 1400})
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto(origin + prefix + "/", wait_until="load")
    lines = page.evaluate("[...new Set(dashboard.visiblePoints.map(p=>p.line))]")
    target = isolated_target(page, lines[0])
    other = isolated_target(page, lines[1])
    points = {point["id"]: point for point in payload["points"]}
    page.mouse.click(target["x"], target["y"])
    for channel in ["ER", "RNA", "AGP"]:
        page.locator(f"#channel-toggle-{channel}").uncheck()
    page.mouse.move(other["x"], other["y"])
    expect(page.locator("#hover-preview")).to_be_visible()
    chosen = ["DNA", "Mito"]
    for selector, point in [("#crop-area canvas", points[target["id"]]),
                            ("#hover-preview canvas", points[other["id"]])]:
        actual = wait_mix(page, selector, point, chosen)
        expected = expected_pixel(bundle, point, chosen)
        assert all(abs(a-b) <= len(chosen) for a, b in zip(actual, expected)), (
            selector, actual, expected)
    expect(page.locator("#cell-id")).to_have_text(target["id"])
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    assert not any(url.endswith("composite.jpg") for url in requests)
    assert not any(url.endswith(".duckdb") for url in requests), "Viewer must not download a database to render images"
    page.locator("details.provenance summary").click()
    expect(page.locator("#catalog-download")).to_be_visible()
    catalog = manifest["catalog"]
    response = page.request.get(origin + prefix + "/" + catalog["path"], timeout=20000)
    assert response.status == 200
    assert len(response.body()) == catalog["bytes"]
    assert hashlib.sha256(response.body()).hexdigest() == catalog["sha256"]
    if os.environ.get("PCCMA_DASHBOARD_EVIDENCE") and prefix:
        directory = Path(os.environ["PCCMA_DASHBOARD_EVIDENCE"])
        directory.mkdir(parents=True, exist_ok=True)
        page.locator("details.provenance summary").click()
        page.evaluate("window.scrollTo(0, 0)")
        other = isolated_target(page, lines[1])
        page.mouse.move(0, 0)
        page.mouse.move(other["x"], other["y"])
        expect(page.locator("#hover-preview")).to_be_visible()
        expect(page.locator("#hover-preview canvas")).to_have_count(1)
        page.screenshot(path=str(directory / "live-mix-comparison.png"), full_page=False)
    assert not errors
    page.close()

