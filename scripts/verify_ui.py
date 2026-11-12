"""Playwright visual verification for the web UI (mandatory for UI work).

Starts the FastAPI app on a free port, drives a real browser (system Chrome
when present, otherwise the Playwright-bundled chromium), and checks both
pages: no uncaught JS errors, no raw script/CSS leaked into the page text,
key elements present, and the quant screen stage actually renders rows.

Then it starts a deterministic fixture app and local mock OpenAI-compatible
server. Synthetic prices and fixture Selection agent contracts keep the full
funnel fast; the mock endpoint exercises the real direct LLM-Black-Litterman
path without network or a live model. The browser verifies both pages, live
progress, result rendering, and native form validation for every strategy.

Screenshots land in temp/ui_verify_*.png — inspect them before declaring an
UI task done. Exits non-zero on any failure.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "temp"
CHROME = "/usr/bin/google-chrome"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.3)
    raise SystemExit(f"FAIL: server on port {port} did not become ready")


def _launch(p):
    if os.path.exists(CHROME):
        return p.chromium.launch(executable_path=CHROME)
    return p.chromium.launch()


def _wait_for_completed_progress_bar(page) -> None:
    page.wait_for_function(
        """() => {
            const panel = document.querySelector('#job-progress');
            const track = document.querySelector('#job-progress-track');
            const fill = track && track.firstElementChild;
            return panel && panel.dataset.status === 'completed' && fill &&
                fill.getBoundingClientRect().width >= track.getBoundingClientRect().width - 1;
        }""",
        timeout=5000,
    )


def _check_page(page, name: str, required_ids: list[str]) -> None:
    page.goto(f"http://127.0.0.1:{_check_page.port}{name}", wait_until="networkidle")
    text = page.inner_text("body")
    for leak in ("function applyTheme", "getElementById", "async function", "document.createElement"):
        if leak in text:
            raise SystemExit(f"FAIL[{name}]: raw script leaked into page text ({leak!r})")
    for cid in required_ids:
        if page.locator(cid).count() == 0:
            raise SystemExit(f"FAIL[{name}]: element {cid} not found")
    page.screenshot(path=str(SHOTS / f"ui_verify_{name.strip('/').replace('/', '_') or 'index'}.png"), full_page=True)
    print(f"OK [{name}]: no leaked script, elements present")


def _start_server(app: str, port: int, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_server(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()


STRATEGY_EXPECTED_LABEL = {
    "sharpe": "Mean-Variance",
    "min_variance": "Minimal Variance",
    "kelly": "Growth",
    "llm_bl": "Black-Litterman",
}


def _check_optimization_strategies(page, port: int) -> None:
    page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
    page.wait_for_timeout(500)
    responses: list[tuple[str, int]] = []
    page.on(
        "response",
        lambda r: responses.append((r.request.method, r.status))
        if r.url.endswith("/optimize")
        else None,
    )
    for strategy, expected in STRATEGY_EXPECTED_LABEL.items():
        page.check(f"input[name=strategy][value={strategy}]")
        if strategy == "llm_bl":
            page.select_option("#llm_view_engine", "direct")
            page.fill("#llm_samples", "2")
        invalid = page.eval_on_selector_all(
            "#optimizer-form [name]",
            "els => els.filter(el => !el.checkValidity())"
            ".map(el => el.name + ': ' + el.validationMessage)",
        )
        if invalid:
            raise SystemExit(
                f"FAIL[{strategy}]: form has invalid controls, native submit is "
                f"blocked: {invalid}"
            )
        page.click("#btn-run")
        try:
            page.wait_for_selector("#results.visible", timeout=60000)
        except Exception:
            raise SystemExit(
                f"FAIL[{strategy}]: clicking Run never rendered results "
                f"(submit blocked or request failed)"
            )
        optimize_posts = [s for m, s in responses if m == "POST"]
        if not optimize_posts or optimize_posts[-1] != 200:
            raise SystemExit(
                f"FAIL[{strategy}]: expected a completed POST /optimize (200), got {optimize_posts}"
            )
        if page.locator("#job-progress").get_attribute("data-status") != "completed":
            raise SystemExit(f"FAIL[{strategy}]: live progress did not reach completed state")
        _wait_for_completed_progress_bar(page)
        if page.locator("#job-progress-events li").count() == 0:
            raise SystemExit(f"FAIL[{strategy}]: live progress event log is empty")
        label = page.inner_text("#result-strategy")
        if expected not in label:
            raise SystemExit(
                f"FAIL[{strategy}]: result label {label!r} does not contain {expected!r}"
            )
        weight_rows = page.locator("#weights-body tr").count()
        if weight_rows == 0:
            raise SystemExit(f"FAIL[{strategy}]: weights table is empty")
        if strategy == "llm_bl":
            view_rows = page.locator("#llm-views-body tr").count()
            if view_rows == 0:
                raise SystemExit("FAIL[llm_bl]: mock endpoint produced no LLM view rows")
            if "degraded" in page.inner_text("#llm-views-note").lower():
                raise SystemExit("FAIL[llm_bl]: mock endpoint unexpectedly degraded")
        page.screenshot(path=str(SHOTS / f"ui_verify_strategy_{strategy}.png"), full_page=True)
        print(
            f"OK [/optimize {strategy}]: POST 200, results rendered "
            f"({label!r}, {weight_rows} weight rows)"
        )


def _check_selection_funnel(page, port: int) -> None:
    page.goto(f"http://127.0.0.1:{port}/select", wait_until="domcontentloaded")
    page.click("#btn-stage-screen")
    page.wait_for_selector("#sectors-body tr", timeout=15000)
    page.wait_for_selector('#job-progress[data-status="completed"]', timeout=15000)
    _wait_for_completed_progress_bar(page)
    if page.locator("#job-progress-events li").count() == 0:
        raise SystemExit("FAIL[/select]: live progress event log is empty")
    screen_rows = page.locator("#sectors-body tr").count()
    if screen_rows != 11:
        raise SystemExit(f"FAIL[/select]: expected 11 fixture sector rows, got {screen_rows}")
    page.screenshot(path=str(SHOTS / "ui_verify_select_screened.png"), full_page=True)

    page.click("#btn-stage-all")
    page.wait_for_selector("#briefing-section:not([hidden])", timeout=15000)
    page.wait_for_selector("#selection-body tr", timeout=15000)
    page.wait_for_selector('#job-progress[data-status="completed"]', timeout=15000)
    _wait_for_completed_progress_bar(page)
    if page.locator("#sectors-body tr").count() != 11:
        raise SystemExit("FAIL[/select]: full funnel did not render all sector rows")
    if page.locator("#selection-body tr").count() == 0:
        raise SystemExit("FAIL[/select]: full funnel produced no selected names")
    if page.locator("#stage-status .warn").count() > 0:
        raise SystemExit("FAIL[/select]: mock full funnel unexpectedly degraded")
    if "Mock market conditions" not in page.inner_text("#briefing-summary"):
        raise SystemExit("FAIL[/select]: fixture briefing did not render")
    page.screenshot(path=str(SHOTS / "ui_verify_select_full_funnel.png"), full_page=True)
    print(
        f"OK [/select]: quant screen + mock full funnel rendered "
        f"({screen_rows} sectors, {page.locator('#selection-body tr').count()} names)"
    )


def main() -> None:
    SHOTS.mkdir(exist_ok=True)
    mock_port = _free_port()
    mock_server = None
    server = None
    fixture_server = None
    try:
        mock_server = _start_server("scripts.mock_openai_server:app", mock_port)
        _wait_ready(mock_port)

        port = _free_port()
        server = _start_server("web.main:app", port)
        fixture_port = _free_port()
        fixture_env = os.environ.copy()
        fixture_env.update({
            "LLM_BASE_URL": f"http://127.0.0.1:{mock_port}/v1",
            "LLM_API_KEY": "mock-key",
            "LLM_MODEL": f"portfolio-mock-ui-{time.time_ns()}",
            "LLM_TIMEOUT": "10",
            "WEBSEARCH_MCP_COMMAND": "",
        })
        fixture_server = _start_server("tests.fixture_app:app", fixture_port, env=fixture_env)
        _wait_ready(port)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            pageerrors: list[str] = []
            page.on("pageerror", lambda e: pageerrors.append(str(e)))

            _check_page.port = port
            _check_page(page, "/", ["#shares", "#optimizer-form", ".page-nav", "#job-progress", "#job-progress-events"])
            _check_page(page, "/select", ["#selection_top_n", "#btn-stage-screen", "#sectors-body", "#selection-body", ".page-nav", "#job-progress", "#job-progress-events"])

            browser.close()
            if pageerrors:
                raise SystemExit("FAIL: uncaught JS errors on production pages: " + " | ".join(pageerrors))

        _wait_ready(fixture_port)
        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            fixture_errors: list[str] = []
            page.on("pageerror", lambda e: fixture_errors.append(str(e)))

            _check_selection_funnel(page, fixture_port)
            _check_optimization_strategies(page, fixture_port)

            browser.close()
            if fixture_errors:
                raise SystemExit("FAIL: uncaught JS errors on fixture runs: " + " | ".join(fixture_errors))
            print("OK: all optimizer strategies, including mock LLM-Black-Litterman, ran in the browser")
        print("\nPASS — screenshots in temp/:")
        for f in sorted(SHOTS.glob("ui_verify_*.png")):
            print(f"  {f}")
    finally:
        for process in (fixture_server, server, mock_server):
            if process is not None:
                _stop_server(process)


if __name__ == "__main__":
    main()
