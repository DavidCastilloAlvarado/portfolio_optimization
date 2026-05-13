"""FastAPI web UI for portfolio optimization."""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
from config.defaults import Config
from data_loader import bulk_stocks, prepare_returns
from optimizer import optimize
from simulation import run_backtest

app = FastAPI(title="Portfolio Optimizer", version="1.0.0")

# ── CSS (VS Code Dark+ theme) ────────────────────────────────────────
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Consolas, 'Courier New', monospace;
    background-color: #1e1e1e;
    color: #d4d4d4;
    min-height: 100vh;
}
.container {
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 24px 64px;
}
h1 {
    font-size: 28px;
    font-weight: 400;
    color: #4ec9b0;
    margin-bottom: 6px;
}
.subtitle {
    color: #808080;
    font-size: 13px;
    margin-bottom: 36px;
}

/* ── Collapsible sections ─────────────────────────────────────────── */
.collapse-section {
    margin-bottom: 10px;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    overflow: hidden;
}
.collapse-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #252526;
    padding: 12px 16px;
    cursor: pointer;
    user-select: none;
    transition: background-color 0.15s;
}
.collapse-header:hover {
    background-color: #2a2d2e;
}
.collapse-header .title {
    font-size: 13px;
    font-weight: 600;
    color: #569cd6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.collapse-header .title .icon {
    font-size: 15px;
}
.collapse-header .chevron {
    color: #808080;
    font-size: 12px;
    transition: transform 0.2s;
}
.collapse-section.open .chevron {
    transform: rotate(90deg);
}
.collapse-body {
    display: none;
    padding: 16px;
    background-color: #1e1e1e;
}
.collapse-section.open .collapse-body {
    display: block;
}

/* ── Form elements ────────────────────────────────────────────────── */
.form-row {
    display: flex;
    gap: 16px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}
.form-group {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 180px;
}
.form-group.full {
    flex: 1 1 100%;
}
.form-group.half {
    flex: 1 1 45%;
}
.form-group.third {
    flex: 1 1 30%;
}
label {
    font-size: 11px;
    color: #969696;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
input[type="text"],
input[type="number"],
textarea,
select {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    color: #d4d4d4;
    padding: 9px 12px;
    font-family: inherit;
    font-size: 13px;
    border-radius: 3px;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
}
input[type="text"]:focus,
input[type="number"]:focus,
textarea:focus,
select:focus {
    border-color: #007acc;
    box-shadow: 0 0 0 1px #007acc;
}
textarea {
    resize: vertical;
    min-height: 48px;
}
select {
    cursor: pointer;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    opacity: 0.5;
}

/* ── Checkbox toggle ─────────────────────────────────────────────── */
.toggle-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding: 10px 14px;
    background-color: #252526;
    border-radius: 3px;
    border: 1px solid #3c3c3c;
}
.toggle-row input[type="checkbox"] {
    width: 18px;
    height: 18px;
    accent-color: #007acc;
    cursor: pointer;
}
.toggle-row label {
    text-transform: none;
    letter-spacing: normal;
    cursor: pointer;
    margin-bottom: 0;
    font-size: 13px;
    color: #d4d4d4;
}
.toggle-row .hint {
    font-size: 11px;
    color: #808080;
    margin-left: auto;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.btn-run {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    padding: 12px 36px;
    font-size: 14px;
    font-family: inherit;
    cursor: pointer;
    border-radius: 3px;
    transition: background-color 0.15s, opacity 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.btn-run:hover:not(:disabled) {
    background-color: #1177bb;
}
.btn-run:active:not(:disabled) {
    background-color: #0d5a8f;
}
.btn-run:disabled {
    background-color: #3c3c3c;
    color: #666666;
    cursor: not-allowed;
}
.btn-secondary {
    background: none;
    border: 1px solid #3c3c3c;
    color: #808080;
    padding: 8px 16px;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
    border-radius: 3px;
    transition: border-color 0.15s, color 0.15s;
}
.btn-secondary:hover {
    border-color: #555555;
    color: #d4d4d4;
}
.btn-group {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}

/* ── Spinner overlay ─────────────────────────────────────────────── */
.spinner-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(30, 30, 30, 0.85);
    z-index: 1000;
    justify-content: center;
    align-items: center;
}
.spinner-overlay.visible {
    display: flex;
}
.spinner {
    width: 48px;
    height: 48px;
    border: 3px solid #3c3c3c;
    border-top-color: #007acc;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.spinner-text {
    color: #d4d4d4;
    font-size: 14px;
    margin-top: 16px;
    text-align: center;
}
.spinner-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* ── Results ─────────────────────────────────────────────────────── */
.results {
    margin-top: 40px;
    display: none;
    animation: fadeIn 0.3s ease;
}
.results.visible {
    display: block;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.results-title {
    font-size: 20px;
    font-weight: 400;
    color: #dcdcaa;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid #3c3c3c;
}
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.result-card {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 16px 18px;
    transition: border-color 0.15s;
}
.result-card:hover {
    border-color: #555555;
}
.result-card .label {
    font-size: 10px;
    color: #808080;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.result-card .value {
    font-size: 22px;
    font-weight: 600;
}
.result-card .value.positive { color: #4ec9b0; }
.result-card .value.negative { color: #f44747; }
.result-card .value.accent { color: #569cd6; }
.result-card .value.warning { color: #dcdcaa; }
.result-card .sub {
    font-size: 11px;
    color: #808080;
    margin-top: 4px;
}

/* ── Weight bars ─────────────────────────────────────────────────── */
.weights-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 24px;
}
.weights-table th {
    text-align: left;
    font-size: 10px;
    color: #808080;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    border-bottom: 1px solid #3c3c3c;
}
.weights-table td {
    padding: 10px 12px;
    font-size: 13px;
    border-bottom: 1px solid #2d2d2d;
}
.weights-table tr:hover td {
    background-color: #2a2d2e;
}
.weights-table .ticker {
    color: #4ec9b0;
    font-weight: 600;
    font-size: 14px;
}
.weights-table .bar-cell {
    width: 30%;
}
.weights-table .bar {
    height: 6px;
    background-color: #0e639c;
    border-radius: 3px;
    transition: width 0.5s ease;
    min-width: 2px;
}
.weights-table .pct {
    color: #dcdcaa;
    text-align: right;
    font-weight: 600;
    font-size: 13px;
}
.weights-table .usd {
    color: #569cd6;
    text-align: right;
    font-family: Consolas, monospace;
    font-size: 13px;
}

/* ── JSON block ──────────────────────────────────────────────────── */
.json-block {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 18px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
    color: #ce9178;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
    margin-top: 16px;
    display: none;
    line-height: 1.6;
}
.json-block.visible {
    display: block;
}
.json-key { color: #9cdcfe; }
.json-string { color: #ce9178; }
.json-number { color: #b5cea8; }
.json-bool { color: #569cd6; }
.json-null { color: #569cd6; }

/* ── Error ───────────────────────────────────────────────────────── */
.error-msg {
    background-color: #3c1e1e;
    border: 1px solid #f44747;
    color: #f44747;
    padding: 14px 18px;
    border-radius: 4px;
    font-size: 13px;
    margin-top: 20px;
    display: none;
}
.error-msg.visible {
    display: block;
}

/* ── Responsive ──────────────────────────────────────────────────── */
@media (max-width: 640px) {
    .container { padding: 20px 12px 48px; }
    .form-group.half, .form-group.third { flex: 1 1 100%; }
    .results-grid { grid-template-columns: 1fr 1fr; }
    .weights-table .bar-cell { display: none; }
}
"""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Optimizer</title>
__CSS__
</head>
<body>
<div class="container">
    <h1>Portfolio Optimizer</h1>
    <p class="subtitle">Modern Portfolio Theory &mdash; Mean-Variance &amp; Minimal Variance</p>

    <form id="optimizer-form">

        <!-- Data -->
        <div class="collapse-section open" onclick="toggleCollapse(this)">
            <div class="collapse-header">
                <span class="title"><span class="icon">&#x1F4CA;</span> Data</span>
                <span class="chevron">&#x25B6;</span>
            </div>
            <div class="collapse-body">
                <div class="form-row">
                    <div class="form-group half">
                        <label for="resample">Resample Frequency</label>
                        <select id="resample" name="resample">
                            <option value="none">Daily</option>
                            <option value="week">Weekly</option>
                            <option value="month">Monthly</option>
                        </select>
                    </div>
                    <div class="form-group half">
                        <label for="days">Historical Days</label>
                        <input type="number" id="days" name="days" value="720" min="30" max="3650">
                    </div>
                </div>
            </div>
        </div>

        <!-- Assets -->
        <div class="collapse-section open" onclick="toggleCollapse(this)">
            <div class="collapse-header">
                <span class="title"><span class="icon">&#x1F3E2;</span> Assets</span>
                <span class="chevron">&#x25B6;</span>
            </div>
            <div class="collapse-body">
                <div class="form-group full">
                    <label for="shares">Tickers (comma-separated)</label>
                    <textarea id="shares" name="shares" rows="2">XLU, QQQ, SCHD, GLDM, SPY, AAPL, TSM, AMD, GOOG</textarea>
                </div>
            </div>
        </div>

        <!-- Optimization -->
        <div class="collapse-section open" onclick="toggleCollapse(this)">
            <div class="collapse-header">
                <span class="title"><span class="icon">&#x2699;</span> Optimization</span>
                <span class="chevron">&#x25B6;</span>
            </div>
            <div class="collapse-body">
                <div class="form-row">
                    <div class="form-group half">
                        <label for="w_limits">Weight Limits (min,max)</label>
                        <input type="text" id="w_limits" name="w_limits" value="0.02,0.12">
                    </div>
                    <div class="form-group half">
                        <label for="risk_free_annual_perc">Risk-Free Rate (%)</label>
                        <input type="number" id="risk_free_annual_perc" name="risk_free_annual_perc" value="5" step="0.1" min="0">
                    </div>
                </div>
                <div class="toggle-row">
                    <input type="checkbox" id="min_variance" name="min_variance" value="on">
                    <label for="min_variance">Minimal Variance</label>
                    <span class="hint">Minimizes risk instead of maximizing Sharpe ratio</span>
                </div>
            </div>
        </div>

        <!-- Investment -->
        <div class="collapse-section open" onclick="toggleCollapse(this)">
            <div class="collapse-header">
                <span class="title"><span class="icon">&#x1F4B0;</span> Investment</span>
                <span class="chevron">&#x25B6;</span>
            </div>
            <div class="collapse-body">
                <div class="form-row">
                    <div class="form-group half">
                        <label for="monto_usd">Initial Investment (USD)</label>
                        <input type="number" id="monto_usd" name="monto_usd" value="10000" min="0">
                    </div>
                    <div class="form-group half">
                        <label for="monthly_delta">Monthly Delta / DCA (USD)</label>
                        <input type="number" id="monthly_delta" name="monthly_delta" value="300" min="0">
                    </div>
                </div>
            </div>
        </div>

        <!-- Simulation -->
        <div class="collapse-section open" onclick="toggleCollapse(this)">
            <div class="collapse-header">
                <span class="title"><span class="icon">&#x1F50D;</span> Simulation</span>
                <span class="chevron">&#x25B6;</span>
            </div>
            <div class="collapse-body">
                <div class="form-row">
                    <div class="form-group half">
                        <label for="sim_days">Simulation Days</label>
                        <input type="number" id="sim_days" name="sim_days" value="252" min="1">
                    </div>
                </div>
            </div>
        </div>

        <div class="btn-group">
            <button type="submit" class="btn-run" id="btn-run">&#x27A1; Run Optimization</button>
            <button type="button" class="btn-secondary" id="btn-reset">&#x21BA; Reset</button>
        </div>
    </form>

    <!-- Results -->
    <div class="results" id="results">
        <div class="results-title">&#x1F4C8; Optimization Results</div>
        <div class="result-card" style="margin-bottom:20px;">
            <div class="label">Strategy</div>
            <div class="value accent" id="result-strategy"></div>
        </div>

        <table class="weights-table">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Allocation</th>
                    <th class="pct">Weight</th>
                    <th class="usd">USD</th>
                </tr>
            </thead>
            <tbody id="weights-body"></tbody>
        </table>

        <div class="results-grid" id="portfolio-stats"></div>

        <!-- Backtest -->
        <div id="backtest-section" style="display:none; margin-top:32px; padding-top:24px; border-top:1px solid #3c3c3c;">
            <div style="font-size:16px; font-weight:400; color:#dcdcaa; margin-bottom:16px;">&#x1F4CA; Backtest Results</div>
            <div class="results-grid" id="backtest-stats"></div>
        </div>

        <div class="btn-group">
            <button class="btn-secondary" id="toggle-json">&#x1F4E6; Show JSON</button>
            <button class="btn-secondary" id="copy-btn" style="display:none;">&#x1F4CB; Copy</button>
        </div>
        <pre class="json-block" id="json-block"></pre>
    </div>

    <div class="error-msg" id="error-msg"></div>
</div>

<!-- Spinner -->
<div class="spinner-overlay" id="spinner">
    <div class="spinner-wrapper">
        <div class="spinner"></div>
        <div class="spinner-text" id="spinner-text">Loading...</div>
    </div>
</div>

<script>
const form = document.getElementById('optimizer-form');
const resultsDiv = document.getElementById('results');
const errorMsg = document.getElementById('error-msg');
const btnRun = document.getElementById('btn-run');
const spinner = document.getElementById('spinner');
const spinnerText = document.getElementById('spinner-text');
const jsonBlock = document.getElementById('json-block');
const copyBtn = document.getElementById('copy-btn');

// ── Collapse toggle ────────────────────────────────────────────────
function toggleCollapse(el) {
    el.classList.toggle('open');
}

// ── Reset form ─────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', () => {
    form.reset();
    resultsDiv.classList.remove('visible');
    jsonBlock.classList.remove('visible');
    copyBtn.style.display = 'none';
    errorMsg.classList.remove('visible');
});

// ── Collapsible sections ───────────────────────────────────────────
document.querySelectorAll('.collapse-header').forEach(header => {
    header.addEventListener('click', (e) => {
        e.stopPropagation();
        header.parentElement.classList.toggle('open');
    });
});

// ── Form submit ────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorMsg.classList.remove('visible');
    resultsDiv.classList.remove('visible');
    jsonBlock.classList.remove('visible');
    copyBtn.style.display = 'none';
    btnRun.disabled = true;
    btnRun.innerHTML = '&#x231B; Running...';
    spinnerText.textContent = 'Fetching data...';
    spinner.classList.add('visible');

    const fd = new FormData(form);
    const mv = fd.get('min_variance');
    if (!mv) fd.delete('min_variance');

    try {
        spinnerText.textContent = 'Optimizing portfolio...';
        const res = await fetch('/optimize', {
            method: 'POST',
            body: fd,
        });

        if (!res.ok) {
            const err = await res.text();
            throw new Error(err);
        }

        const data = await res.json();
        renderResults(data);
        resultsDiv.classList.add('visible');
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        errorMsg.textContent = 'Error: ' + (err.message || 'Unknown error');
        errorMsg.classList.add('visible');
    } finally {
        btnRun.disabled = false;
        btnRun.innerHTML = '&#x27A1; Run Optimization';
        spinner.classList.remove('visible');
    }
});

// ── Render results ─────────────────────────────────────────────────
function renderResults(data) {
    document.getElementById('result-strategy').textContent = data.strategy || '';

    // Weights table
    const tbody = document.getElementById('weights-body');
    tbody.innerHTML = '';
    if (data.weights) {
        const maxW = Math.max(...data.weights.map(w => w.weight_pct));
        for (const w of data.weights) {
            const tr = document.createElement('tr');
            const barW = maxW > 0 ? (w.weight_pct / maxW * 100) : 0;
            tr.innerHTML = `
                <td class="ticker">${w.ticker}</td>
                <td class="bar-cell"><div class="bar" style="width:${barW}%"></div></td>
                <td class="pct">${w.weight_pct.toFixed(2)}%</td>
                <td class="usd">$${w.usd.toFixed(2)}</td>
            `;
            tbody.appendChild(tr);
        }
    }

    // Portfolio stats
    const statsDiv = document.getElementById('portfolio-stats');
    statsDiv.innerHTML = '';
    const retPct = data.portfolio_return_pct || 0;
    const stats = [
        { label: 'Expected Return (annualized)', value: retPct.toFixed(2) + '%', cls: retPct >= 0 ? 'positive' : 'negative', sub: 'Daily mean * 365' },
        { label: 'Std Deviation (annualized)', value: (data.portfolio_std_pct || 0).toFixed(2) + '%', cls: 'accent', sub: 'Daily std * sqrt(365)' },
        { label: 'Sharpe Ratio', value: (data.sharpe_ratio || 0).toFixed(4), cls: 'warning', sub: '(Return - RF) / StdDev' },
    ];
    for (const s of stats) {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="label">${s.label}</div>
            <div class="value ${s.cls}">${s.value}</div>
            <div class="sub">${s.sub}</div>
        `;
        statsDiv.appendChild(card);
    }

    // Backtest
    const btSection = document.getElementById('backtest-section');
    const btStatsDiv = document.getElementById('backtest-stats');
    btStatsDiv.innerHTML = '';

    if (data.backtest) {
        btSection.style.display = 'block';
        const lump = data.backtest.lump_sum;
        if (lump) {
            const cards = [
                { label: 'Final Value', value: '$' + (lump.final_value || 0).toFixed(2), cls: 'positive' },
                { label: 'Total Return', value: ((lump.total_return_pct || 0) >= 0 ? '+' : '') + (lump.total_return_pct || 0).toFixed(2) + '%', cls: (lump.total_return_pct || 0) >= 0 ? 'positive' : 'negative' },
                { label: 'Max Drawdown', value: '-' + (lump.max_drawdown_pct || 0).toFixed(2) + '%', cls: 'negative' },
                { label: 'Sharpe', value: (lump.sharpe_annualized || 0).toFixed(4), cls: 'warning' },
                { label: 'Period', value: (lump.trading_days || 0) + ' days', cls: 'accent' },
            ];
            for (const c of cards) {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `<div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div>`;
                btStatsDiv.appendChild(card);
            }
        }

        const dca = data.backtest.dca;
        if (dca) {
            const cards = [
                { label: 'Final Value', value: '$' + (dca.final_value || 0).toFixed(2), cls: 'positive' },
                { label: 'Total Invested', value: '$' + (dca.total_invested || 0).toFixed(2), cls: 'accent' },
                { label: 'Total Return', value: ((dca.total_return_pct || 0) >= 0 ? '+' : '') + (dca.total_return_pct || 0).toFixed(2) + '%', cls: (dca.total_return_pct || 0) >= 0 ? 'positive' : 'negative' },
                { label: 'Sharpe', value: (dca.sharpe_annualized || 0).toFixed(4), cls: 'warning' },
                { label: 'Contributions', value: '$' + (dca.monthly_addition || 0) + '/mo x ' + (dca.months_contributed || 0) + ' mo', cls: 'accent' },
            ];
            for (const c of cards) {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `<div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div>`;
                btStatsDiv.appendChild(card);
            }
        }
    } else {
        btSection.style.display = 'none';
    }

    // JSON with syntax highlighting
    const raw = JSON.stringify(data, null, 2);
    jsonBlock.innerHTML = syntaxHighlight(raw);
    copyBtn.style.display = 'inline-flex';
}

// ── JSON syntax highlight ──────────────────────────────────────────
function syntaxHighlight(json) {
    return json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
        .replace(/: "(.*?)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/: (-?\\d+\\.?\\d*e?[+-]?\\d*)/g, ': <span class="json-number">$1</span>')
        .replace(/: (true|false)/g, ': <span class="json-bool">$1</span>')
        .replace(/: (null)/g, ': <span class="json-null">$1</span>');
}

// ── Toggle JSON ────────────────────────────────────────────────────
document.getElementById('toggle-json').addEventListener('click', () => {
    const visible = jsonBlock.classList.toggle('visible');
    document.getElementById('toggle-json').innerHTML = visible ? '&#x1F4E5; Hide JSON' : '&#x1F4E6; Show JSON';
});

// ── Copy ───────────────────────────────────────────────────────────
copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(jsonBlock.textContent).then(() => {
        copyBtn.innerHTML = '&#x2705; Copied!';
        setTimeout(() => { copyBtn.innerHTML = '&#x1F4CB; Copy'; }, 1500);
    });
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    return HTML.replace("__CSS__", f"<style>\n{CSS}\n</style>")


@app.post("/optimize")
async def optimize_endpoint(
    resample: str = Form("none"),
    days: str = Form("720"),
    shares: str = Form(""),
    w_limits: str = Form("0.02,0.12"),
    min_variance: str = Form("off"),
    monto_usd: str = Form("10000"),
    monthly_delta: str = Form("300"),
    sim_days: str = Form("252"),
    risk_free_annual_perc: str = Form("5"),
):
    """Run the optimization pipeline with form parameters."""
    data = {
        "resample": resample,
        "days": days,
        "shares": shares,
        "w_limits": w_limits,
        "min_variance": min_variance,
        "monto_usd": monto_usd,
        "monthly_delta": monthly_delta,
        "sim_days": sim_days,
        "risk_free_annual_perc": risk_free_annual_perc,
    }

    cfg = Config.from_dict(data)

    # Data loading
    resample_val = cfg.get_resample()
    raw_data = bulk_stocks(cfg.shares, cfg.days)
    data_df, returns = prepare_returns(raw_data, resample=resample_val)

    names = data_df.columns.tolist()
    mean_returns = np.array(returns.mean())
    cov_returns = np.array(returns.cov())

    # Optimization
    weights, port_mean_val, port_std, strategy = optimize(
        mean_returns.copy(), cov_returns.copy(), cfg.risk_free, cfg.w_limits, cfg.min_variance,
    )

    # Build weights response
    weights_resp = []
    for name, w in zip(names, weights):
        weights_resp.append({
            "ticker": name,
            "weight_pct": float(w * 100),
            "usd": float(w * cfg.monto_usd),
        })

    # Sharpe ratio
    daily_rf = cfg.risk_free
    port_return_daily = float(port_mean_val)
    port_std_daily = float(port_std)
    if port_std_daily > 0:
        sharpe_daily = (port_return_daily - daily_rf) / port_std_daily
        sharpe_annual = sharpe_daily * np.sqrt(365)
    else:
        sharpe_annual = 0.0

    result = {
        "strategy": strategy,
        "weights": weights_resp,
        "portfolio_return_pct": float(port_return_daily * 365 * 100),
        "portfolio_std_pct": float(port_std_daily * np.sqrt(365) * 100),
        "sharpe_ratio": float(sharpe_annual),
    }

    # Backtest
    backtest = run_backtest(
        cfg.monto_usd, cfg.shares, weights, cfg.sim_days,
        cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
    )
    if backtest and "error" not in backtest:
        result["backtest"] = backtest

    return result
