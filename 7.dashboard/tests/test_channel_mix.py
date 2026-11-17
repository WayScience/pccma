"""Pixel-level browser checks of every grayscale-channel combination.

Only the small synthetic fixture is used here; real pilot coverage is separate.
"""

from itertools import combinations
import re
from urllib.parse import unquote

from PIL import Image
from playwright.sync_api import expect

from test_browser import browser, origin, pointer_target, site  # noqa: F401
from test_build_report import tables  # noqa: F401


COLORS = {
    "DNA": (77, 124, 255), "ER": (58, 214, 176), "RNA": (255, 209, 102),
    "AGP": (255, 95, 162), "Mito": (255, 107, 77),
}


def expected_pixel(bundle, point, chosen):
    """Independent additive-color oracle using the exact decoded grayscale JPEGs."""
    values = []
    for channel in chosen:
        with Image.open(bundle / unquote(point["images"][channel])) as image:
            values.append((image.getpixel((112, 112)), COLORS[channel]))
    return [min(255, sum(round(value * color[component] / 255) for value, color in values))
            for component in range(3)] + [255]


def wait_mix(page, selector, point, chosen):
    label = " + ".join(chosen) + " crop · " + point["id"]
    page.wait_for_function("""({selector,label}) => {
        const canvas=document.querySelector(selector);
        return canvas && canvas.getAttribute('aria-label')===label;
    }""", arg={"selector": selector, "label": label}, timeout=5000)
    return page.locator(selector).evaluate("""canvas => Array.from(
        canvas.getContext('2d').getImageData(112,112,1,1).data)
    """)


def test_all_32_selections_mix_correct_pixels_for_hover_and_pin(browser, origin, site):
    bundle, payload = site
    page = browser.new_page(viewport={"width": 1600, "height": 1400})
    errors, composite_requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: composite_requests.append(request.url)
            if request.url.endswith("_composite.jpg") else None)
    page.goto(origin + "/pccma/report/", wait_until="load")
    # Users can set the shared selection BEFORE hovering/pinning anything.
    expect(page.locator("#channel-toggle-DNA")).to_be_enabled()
    expect(page.locator("#plate-label")).to_contain_text("Timepoint 24 h")
    first = pointer_target(page, 0)
    page.mouse.click(first["x"], first["y"])
    expect(page.locator("#cell-id")).to_have_text(first["id"])
    points = {point["id"]: point for point in payload["points"]}
    pinned = points[first["id"]]
    checked = []
    for count in range(6):
        for selection in combinations(COLORS, count):
            chosen = list(selection)
            for channel in COLORS:
                page.locator(f"#channel-toggle-{channel}").set_checked(channel in chosen)
            second = pointer_target(page, 1)
            page.mouse.move(second["x"], second["y"])
            hovered = points[second["id"]]
            expect(page.locator("#hover-preview")).to_be_visible()
            expect(page.locator("#cell-id")).to_have_text(pinned["id"])
            expect(page.locator("#pin-status")).to_have_text("Pinned")
            if chosen:
                for selector, point in [("#crop-area canvas", pinned),
                                        ("#hover-preview canvas", hovered)]:
                    actual = wait_mix(page, selector, point, chosen)
                    expected = expected_pixel(bundle, point, chosen)
                    # Each canvas multiply/add can round by a byte; tolerate at
                    # most one per layer, not arbitrary visual similarity.
                    assert all(abs(a - b) <= len(chosen) for a, b in zip(actual, expected)), (
                        chosen, selector, actual, expected)
                expect(page.locator("#crop-area img")).to_have_count(len(chosen))
            else:
                expect(page.locator("#crop-area canvas")).to_have_count(0)
                expect(page.locator("#hover-preview canvas")).to_have_count(0)
                expect(page.locator("#crop-area")).to_contain_text("Select at least one channel")
                expect(page.locator("#hover-preview")).to_contain_text("Select at least one channel")
            checked.append(selection)
    assert len(checked) == 32
    assert not composite_requests, "Even all-five mixing must work from grayscale alone"
    page.mouse.move(0, 0)
    expect(page.locator("#hover-preview")).to_be_hidden()
    expect(page.locator("#cell-id")).to_have_text(pinned["id"])
    assert not errors
    page.close()


def test_grayscale_mix_on_direct_file_open_and_mobile(browser, origin, site):
    bundle, _payload = site
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto((bundle / "index.html").as_uri(), wait_until="load")
    page.locator("#next-cell").click()
    expect(page.locator("#crop-area canvas")).to_have_count(1)
    for channel in ["ER", "RNA", "AGP"]:
        page.locator(f"#channel-toggle-{channel}").uncheck()
    expect(page.locator("#crop-area canvas")).to_have_attribute("aria-label", re.compile(r"DNA \+ Mito crop"))
    # file:// canvases may be tainted; rendering/screenshot, not pixel readback,
    # is the contract for the downloadable report.
    assert page.locator("#crop-area canvas").evaluate("c=>c.width===224 && c.height===224")
    assert not errors
    page.close()
    context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    mobile = context.new_page()
    mobile.goto(origin + "/pccma/report/", wait_until="load")
    mobile.locator("#channel-toggle-Mito").uncheck()
    mobile.locator("#next-cell").click()
    expect(mobile.locator("#crop-area canvas")).to_have_count(1)
    expect(mobile.locator("#crop-area img")).to_have_count(4)
    assert mobile.evaluate("document.documentElement.scrollWidth<=innerWidth")
    context.close()
