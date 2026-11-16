"""Real Chromium interaction checks; generated images are test fixtures, not cell crops."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pandas as pd
import pytest
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from test_build_report import builder, tables  # noqa: F401 (shared pytest fixture)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True)
        yield instance
        instance.close()


@pytest.fixture
def origin(tmp_path):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def site(tables, tmp_path):
    """Three distinctly located synthetic fixture cells in two independent groups."""
    umap, index = tables
    first = umap.iloc[[0]].copy()
    second = first.copy()
    second["Metadata_Nuclei_Location_Center_X"] += 10
    second["umap_1"], second["umap_2"] = 3., 4.
    second["Metadata_cell_line"] = "Fixture B"
    third = first.copy()
    third["Metadata_Nuclei_Location_Center_X"] += 20
    third["umap_1"], third["umap_2"] = 900., -700.
    third["cancer_group"] = "Other fixture group"
    umap = pd.concat([first, second, third], ignore_index=True)
    indices = []
    warehouse = tmp_path / "warehouse"
    for i, row in umap.iterrows():
        record = index.iloc[[0]].copy()
        record["Metadata_Nuclei_Location_Center_X"] = row.Metadata_Nuclei_Location_Center_X
        record["Metadata_cell_line"] = row.Metadata_cell_line
        record["Metadata_Nuclei_Number_Object_Number"] = i + 3
        record["Metadata_Cells_Number_Object_Number"] = i + 2
        for channel, col in builder().PATH_COLUMNS.items():
            relative = f"crops/P1/fixture_{i}_{channel}.jpg"
            record[col] = relative
            target = warehouse / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB" if channel == "composite" else "L", (224, 224), (i + 1) * 50).save(target, "JPEG")
        indices.append(record)
    umap_path, index_path = tmp_path / "umap.parquet", tmp_path / "index.parquet"
    umap.to_parquet(umap_path)
    pd.concat(indices).to_parquet(index_path)
    output = tmp_path / "pccma" / "report"
    payload = builder().build_report(umap_path=umap_path, index_path=index_path, warehouse=warehouse,
                                     output=output, plate="P1", run_id="synthetic-test-fixture")
    return output, payload


def pointer_target(page, index=0):
    return page.evaluate("""index => {
        const point = dashboard.visiblePoints[index];
        const screen = dashboard.projection.forward(point);
        const rect = document.getElementById('plot').getBoundingClientRect();
        return {id:point.id, x:screen.x+rect.left, y:screen.y+rect.top};
    }""", index)


def test_hover_pin_channels_and_prefix_paths(browser, origin, site):
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    errors, requests, failed = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.on("response", lambda response: failed.append(response.url) if response.status >= 400 else None)
    page.goto(origin + "/pccma/report/", wait_until="load")
    expect(page.locator("#error-banner")).to_be_hidden()
    expect(page.locator("#image-warning")).to_be_hidden()
    assert page.evaluate("dashboard.visiblePoints.length") == 2
    target = pointer_target(page)
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#cell-id")).to_have_text(target["id"])
    expect(page.locator("#crop-area img")).to_have_count(5)
    assert page.locator("#crop-area img").evaluate_all("images=>images.every(img=>img.complete&&img.naturalWidth===224)")
    assert all("/pccma/report/assets/crops/" in url for url in requests if url.endswith(".jpg"))
    page.mouse.click(target["x"], target["y"])
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    another = pointer_target(page, 1)
    page.mouse.move(another["x"], another["y"])
    expect(page.locator("#cell-id")).to_have_text(target["id"])
    for channel in ["ER", "RNA", "AGP", "Mito"]:
        page.locator(f"#channel-toggle-{channel}").uncheck()
    expect(page.locator("#crop-area img")).to_have_count(1)  # only the DNA thumbnail remains
    mix_canvas = page.locator("#crop-area canvas")
    page.wait_for_function("document.querySelectorAll('#crop-area canvas').length===1")
    # The single-channel mix is still tinted and drawn for real, not a blank canvas.
    assert mix_canvas.evaluate("""canvas => {
        const ctx = canvas.getContext('2d');
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 3] > 0 && pixels[i] > 0) return true;
        return false;
    }"""), "the DNA-only mix canvas must actually paint tinted pixels"
    for channel in ["ER", "RNA"]:
        page.locator(f"#channel-toggle-{channel}").check()
    expect(page.locator("#crop-area img")).to_have_count(3)  # DNA + ER + RNA thumbnails, live mix stays a canvas
    page.wait_for_function("document.querySelectorAll('#crop-area canvas').length===1")
    page.locator("#channel-toggle-DNA").uncheck()
    page.locator("#channel-toggle-ER").uncheck()
    page.locator("#channel-toggle-RNA").uncheck()
    expect(page.locator("#crop-area")).to_contain_text("Select at least one channel")
    page.locator("#channel-toggle-DNA").check()
    page.locator("#channel-toggle-ER").check()
    page.locator("#channel-toggle-RNA").check()
    page.locator("#channel-toggle-AGP").check()
    page.locator("#channel-toggle-Mito").check()
    expect(page.locator("#crop-area img")).to_have_count(5)  # live five-channel mix + all 5 thumbnails
    page.locator("#unpin-cell").click()
    page.locator("#next-cell").click()
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    page.locator("#group-select").select_option("Other fixture group")
    assert page.evaluate("dashboard.visiblePoints.length") == 1
    assert page.evaluate("dashboard.selectedPoint") is None
    expect(page.locator("#inspector-title")).to_have_text("Point preview")
    assert not errors
    assert not failed
    assert all(url.startswith(origin) for url in requests)
    page.close()


def test_mobile_touch_filters_zoom_and_file_open(browser, origin, site):
    context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(origin + "/pccma/report/", wait_until="load")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    point = pointer_target(page)
    page.touchscreen.tap(point["x"], point["y"])
    expect(page.locator("#pin-status")).to_have_text("Pinned")
    page.locator("#zoom-in").click()
    expect(page.locator("#zoom-label")).to_have_text("130%")
    page.locator("#reset-view").click()
    expect(page.locator("#zoom-label")).to_have_text("100%")
    # Hide all categories via actual UI; ensure the pinned cell cannot survive it.
    for button in page.locator("#legend button").all():
        button.click()
    expect(page.locator("#empty-state")).to_be_visible()
    assert page.evaluate("dashboard.selectedPoint") is None
    assert not errors
    context.close()
    file_page = browser.new_page()
    file_page.goto((site[0] / "index.html").as_uri(), wait_until="load")
    file_page.locator("#next-cell").click()
    expect(file_page.locator("#crop-area img")).to_have_count(5)
    assert file_page.locator("#crop-area img").evaluate_all("images=>images.every(img=>img.naturalWidth===224)")
    file_page.close()


def test_image_404_is_visible_and_never_shows_stale_pixels(browser, origin, site):
    page = browser.new_page()
    page.route("**/*_DNA.jpg", lambda route: route.fulfill(status=404, body="missing test fixture"))
    page.goto(origin + "/pccma/report/", wait_until="load")
    page.locator("#next-cell").click()
    expect(page.locator(".image-error")).to_have_count(2)
    expect(page.locator(".image-error").first).to_contain_text("Image could not be loaded")
    assert page.locator("#crop-area img").count() == 4
    page.close()


@pytest.mark.parametrize("viewport", [{"width": 1280, "height": 577}, {"width": 900, "height": 700}])
def test_hover_crop_is_fully_visible_without_scrolling(browser, origin, site, viewport):
    page = browser.new_page(viewport=viewport)
    page.goto(origin + "/pccma/report/", wait_until="load")
    target = pointer_target(page, 1)  # The upper fixture point remains in view.
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#cell-id")).to_have_text(target["id"])
    page.wait_for_function("document.querySelectorAll('#crop-area img').length===5")
    # A decoded image below the fold does not satisfy a visible hover preview.
    assert page.locator("#hover-preview canvas").evaluate_all("""images => images.some(image => {
        const rect=image.getBoundingClientRect();
        return image.width===224 && rect.width>=100 && rect.height>=100 &&
            rect.left>=0 && rect.top>=0 && rect.right<=innerWidth && rect.bottom<=innerHeight;
    })"""), "Hover must display a complete crop inside the current viewport"
    page.close()


def test_real_plate_coordinates_all_groups_without_fake_image_requests(browser, tmp_path, origin):
    root = Path(__file__).resolve().parents[2]
    index = root / "5a.object_image_crops/data/pilot_runs/BR00150695-20261113T155150Z-g790ecb4/crops_index.parquet"
    if not index.exists():
        pytest.skip("Real pilot index is local, not part of a clean checkout")
    output = tmp_path / "pccma" / "real"
    payload = builder().build_report(
        umap_path=root / "6.profile_analysis/results/3.single_cell_embedding/single_cell_umap_coordinates_by_cancer_group.parquet",
        index_path=index, warehouse=None, output=output, plate="BR00150695",
        run_id="BR00150695-20261113T155150Z-g790ecb4", allow_missing_images=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    requests, errors = [], []
    page.on("request", lambda request: requests.append(request))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(origin + "/pccma/real/", wait_until="load")
    expect(page.locator("#image-warning")).to_be_visible()
    assert payload["summary"]["umap_cells"] == 6500
    groups = page.locator("#group-select option").evaluate_all("options=>options.map(option=>option.value)")
    assert len(groups) == 7
    observed = 0
    for group in groups:
        page.locator("#group-select").select_option(group)
        observed += page.evaluate("dashboard.visiblePoints.length")
        point = pointer_target(page)
        page.mouse.move(point["x"], point["y"])
        # Dense real points may choose a closer neighbor at the same screen pixel.
        expect(page.locator("#cell-id")).not_to_have_text("No cell selected")
        assert page.evaluate("dashboard.hoveredPoint.group") == group
        # Confirm rendered colored point pixels, not just a working hit-test index.
        assert page.evaluate("""() => {
            const canvas=document.getElementById('plot');
            const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            let count=0;
            for(let i=0;i<pixels.length;i+=4){
                const low=Math.min(pixels[i],pixels[i+1],pixels[i+2]);
                const high=Math.max(pixels[i],pixels[i+1],pixels[i+2]);
                if(pixels[i+3]>80 && low>50 && high-low>45) count++;
            }
            return count>100;
        }""")
        expect(page.locator("#crop-area")).to_contain_text("Crops unavailable")
    assert observed == 6500
    assert not [request.url for request in requests if request.resource_type == "image"]
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors
    page.close()
