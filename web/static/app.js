
const form = document.getElementById('optimizer-form');
const resultsDiv = document.getElementById('results');
const errorMsg = document.getElementById('error-msg');
const btnRun = document.getElementById('btn-run');
const spinner = document.getElementById('spinner');
const spinnerText = document.getElementById('spinner-text');
const jsonBlock = document.getElementById('json-block');
const copyBtn = document.getElementById('copy-btn');
let lastJson = '';

// ── Theme ────────────────────────────────────────────────────────────
const themeToggle = document.getElementById('theme-toggle');
function applyTheme(name) {
    document.documentElement.setAttribute('data-theme', name);
    themeToggle.innerHTML = name === 'dark' ? '&#x1F319; Dark' : '&#x2600;&#xFE0E; Light';
}
applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');
themeToggle.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('po-theme', next); } catch (e) {}
});

// ── Reset form ─────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', () => {
    form.reset();
    resultsDiv.classList.remove('visible');
    jsonBlock.classList.remove('visible');
    copyBtn.style.display = 'none';
    errorMsg.classList.remove('visible');
    perTickerPanel.hidden = true;
    btnPerTicker.classList.remove('open');
    btnPerTicker.setAttribute('aria-expanded', 'false');
    btnPerTicker.querySelector('.chev').innerHTML = '&#x25B8;';
    perTickerHidden.value = '';
    perTickerValues = {};
});

// ── Collapsible sections ───────────────────────────────────────────
document.querySelectorAll('.collapse-header').forEach(header => {
    header.addEventListener('click', (e) => {
        e.stopPropagation();
        header.parentElement.classList.toggle('open');
    });
});

// ── Per-ticker weight limits ───────────────────────────────────────
const sharesInput = document.getElementById('shares');
const wLimitsInput = document.getElementById('w_limits');
const perTickerPanel = document.getElementById('per-ticker-panel');
const perTickerRows = document.getElementById('per-ticker-rows');
const perTickerHidden = document.getElementById('w_limits_per_ticker');
const btnPerTicker = document.getElementById('btn-per-ticker');
let perTickerValues = {};

function parseTickers(text) {
    const seen = new Set();
    const tickers = [];
    for (const raw of (text || '').split(',')) {
        const t = raw.trim().toUpperCase();
        if (!t || seen.has(t)) continue;
        seen.add(t);
        tickers.push(t);
    }
    return tickers;
}

function parseGlobalLimits() {
    const parts = (wLimitsInput.value || '').split(',').map(x => parseFloat(x.trim()));
    const min = Number.isFinite(parts[0]) ? parts[0] : 0.02;
    const max = Number.isFinite(parts[1]) ? parts[1] : 0.12;
    return {
        min: Math.min(Math.max(min, 0), 1),
        max: Math.min(Math.max(max, 0), 1),
    };
}

function buildPerTickerRows() {
    const tickers = parseTickers(sharesInput.value);
    const g = parseGlobalLimits();
    perTickerRows.innerHTML = '';
    for (const t of tickers) {
        const v = perTickerValues[t] || { min: g.min, max: g.max };
        perTickerValues[t] = v;
        const row = document.createElement('div');
        row.className = 'per-ticker-row';
        row.dataset.ticker = t;
        row.innerHTML = `
            <span class="pt-name">${escapeHtml(t)}</span>
            <div class="pt-bound">
                <input type="range" class="pt-range" min="0" max="1" step="0.01" value="${v.min}">
                <input type="number" class="pt-num" min="0" max="1" step="0.01" value="${v.min}">
            </div>
            <div class="pt-bound">
                <input type="range" class="pt-range" min="0" max="1" step="0.01" value="${v.max}">
                <input type="number" class="pt-num" min="0" max="1" step="0.01" value="${v.max}">
            </div>
        `;
        perTickerRows.appendChild(row);
    }
    if (tickers.length === 0) {
        perTickerRows.innerHTML = '<div class="pt-empty">Add tickers in the Assets section</div>';
    }
}

function setBound(row, idx, value) {
    const ranges = row.querySelectorAll('.pt-range');
    const nums = row.querySelectorAll('.pt-num');
    let v = Math.min(Math.max(value, 0), 1);
    if (idx === 0) v = Math.min(v, parseFloat(ranges[1].value));
    else v = Math.max(v, parseFloat(ranges[0].value));
    ranges[idx].value = v;
    nums[idx].value = v;
    perTickerValues[row.dataset.ticker] = {
        min: parseFloat(ranges[0].value),
        max: parseFloat(ranges[1].value),
    };
}

perTickerRows.addEventListener('input', (e) => {
    const row = e.target.closest('.per-ticker-row');
    if (!row) return;
    if (e.target.classList.contains('pt-range')) {
        const idx = row.querySelector('.pt-range') === e.target ? 0 : 1;
        setBound(row, idx, parseFloat(e.target.value));
    } else if (e.target.classList.contains('pt-num')) {
        const idx = row.querySelectorAll('.pt-num')[0] === e.target ? 0 : 1;
        const v = parseFloat(e.target.value);
        if (Number.isFinite(v)) setBound(row, idx, v);
    }
});

perTickerRows.addEventListener('change', (e) => {
    if (!e.target.classList.contains('pt-num')) return;
    const row = e.target.closest('.per-ticker-row');
    if (!row) return;
    const idx = row.querySelectorAll('.pt-num')[0] === e.target ? 0 : 1;
    const v = parseFloat(e.target.value);
    setBound(row, idx, Number.isFinite(v) ? v : 0);
});

btnPerTicker.addEventListener('click', () => {
    const open = perTickerPanel.hidden;
    perTickerPanel.hidden = !open;
    btnPerTicker.classList.toggle('open', open);
    btnPerTicker.setAttribute('aria-expanded', String(open));
    btnPerTicker.querySelector('.chev').innerHTML = open ? '&#x25BE;' : '&#x25B8;';
    if (open) buildPerTickerRows();
    else perTickerHidden.value = '';
});

sharesInput.addEventListener('input', () => {
    if (!perTickerPanel.hidden) buildPerTickerRows();
});

function syncPerTickerPayload() {
    if (perTickerPanel.hidden) {
        perTickerHidden.value = '';
        return;
    }
    const parts = [];
    for (const row of perTickerRows.querySelectorAll('.per-ticker-row')) {
        let v = perTickerValues[row.dataset.ticker];
        if (!v) {
            const ranges = row.querySelectorAll('.pt-range');
            v = { min: parseFloat(ranges[0].value), max: parseFloat(ranges[1].value) };
        }
        parts.push(row.dataset.ticker + ':' + v.min + ',' + v.max);
    }
    perTickerHidden.value = parts.join(';');
}

function validateWeightLimits() {
    const tickers = parseTickers(sharesInput.value);
    if (tickers.length === 0) return null;
    let minSum = 0;
    let maxSum = 0;
    if (perTickerPanel.hidden) {
        const g = parseGlobalLimits();
        minSum = g.min * tickers.length;
        maxSum = g.max * tickers.length;
    } else {
        for (const t of tickers) {
            const v = perTickerValues[t];
            if (!v) continue;
            minSum += v.min;
            maxSum += v.max;
        }
    }
    if (minSum > 1 + 1e-9) {
        return 'Sum of min limits is ' + (minSum * 100).toFixed(1) + '% — must be 100% or less.';
    }
    if (maxSum < 1 - 1e-9) {
        return 'Sum of max limits is ' + (maxSum * 100).toFixed(1) + '% — must be at least 100%.';
    }
    return null;
}

// ── Form submit ────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const limitsErr = validateWeightLimits();
    if (limitsErr) {
        renderError(limitsErr);
        return;
    }
    errorMsg.classList.remove('visible');
    resultsDiv.classList.remove('visible');
    jsonBlock.classList.remove('visible');
    copyBtn.style.display = 'none';
    btnRun.disabled = true;
    btnRun.innerHTML = '&#x231B; Running...';
    spinnerText.textContent = 'Fetching data...';
    spinner.classList.add('visible');

    syncPerTickerPayload();
    const fd = new FormData(form);
    const mv = fd.get('min_variance');
    if (!mv) fd.delete('min_variance');

    try {
        spinnerText.textContent = 'Fetching market data...';
        const res = await fetch('/optimize', {
            method: 'POST',
            body: fd,
        });

        if (!res.ok) {
            try {
                const errData = await res.json();
                renderError(errData.traceback || errData.error || 'Unknown server error');
            } catch {
                const err = await res.text();
                renderError(err || 'Unknown server error');
            }
            return;
        }

        const data = await res.json();
        if (data.error) {
            renderError(data.error);
            return;
        }
        renderResults(data);
        resultsDiv.classList.add('visible');
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        renderError(err.message || 'Unknown error');
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
        if (data.backtest.error) {
            btSection.style.display = 'block';
            btStatsDiv.innerHTML = '<div class="bt-error">Backtest failed: ' + escapeHtml(data.backtest.error) + '</div>';
        } else {
            btSection.style.display = 'block';
            const lump = data.backtest.lump_sum;
        if (lump) {
            const lumpSection = document.createElement('div');
            lumpSection.className = 'bt-section';
            lumpSection.innerHTML = '<div class="bt-section-title">Lump Sum</div>';
            const lumpCards = document.createElement('div');
            lumpCards.className = 'results-grid';
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
                card.innerHTML = '<div class="label">' + c.label + '</div><div class="value ' + c.cls + '">' + c.value + '</div>';
                lumpCards.appendChild(card);
            }
            lumpSection.appendChild(lumpCards);
            btStatsDiv.appendChild(lumpSection);
        }

        const dca = data.backtest.dca;
        if (dca) {
            const dcaSection = document.createElement('div');
            dcaSection.className = 'bt-section';
            dcaSection.innerHTML = '<div class="bt-section-title">DCA</div>';
            const dcaCards = document.createElement('div');
            dcaCards.className = 'results-grid';
            const cards = [
                { label: 'Final Value', value: '$' + (dca.final_value || 0).toFixed(2), cls: 'positive' },
                { label: 'Total Return', value: ((dca.total_return_pct || 0) >= 0 ? '+' : '') + (dca.total_return_pct || 0).toFixed(2) + '%', cls: (dca.total_return_pct || 0) >= 0 ? 'positive' : 'negative' },
                { label: 'Max Drawdown', value: '-' + (dca.max_drawdown_pct || 0).toFixed(2) + '%', cls: 'negative' },
                { label: 'Sharpe', value: (dca.sharpe_annualized || 0).toFixed(4), cls: 'warning' },
            ];
            for (const c of cards) {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = '<div class="label">' + c.label + '</div><div class="value ' + c.cls + '">' + c.value + '</div>';
                dcaCards.appendChild(card);
            }
            // Total invested card with hover tooltip for contributions
            if (dca.total_invested != null || dca.monthly_addition != null) {
                const tip = '$' + (dca.monthly_addition || 0) + '/mo x ' + (dca.months_contributed || 0) + ' mo';
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = '<div class="label">Total Invested</div><div class="value accent" style="cursor:help;" title="' + tip + '">$' + (dca.total_invested || 0).toFixed(2) + '</div>';
                dcaCards.appendChild(card);
            }
            dcaSection.appendChild(dcaCards);
            btStatsDiv.appendChild(dcaSection);
        }
        }
    } else {
        btSection.style.display = 'none';
    }

    // JSON with syntax highlighting
    lastJson = JSON.stringify(data, null, 2);
    jsonBlock.innerHTML = syntaxHighlight(lastJson);
    copyBtn.style.display = 'inline-flex';
}

// ── JSON syntax highlight ──────────────────────────────────────────
function syntaxHighlight(json) {
    return json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
        .replace(/: "(.*?)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/: (-?\d+\.?\d*e?[+-]?\d*)/g, ': <span class="json-number">$1</span>')
        .replace(/: (true|false)/g, ': <span class="json-bool">$1</span>')
        .replace(/: (null)/g, ': <span class="json-null">$1</span>');
}

// ── Toggle JSON ────────────────────────────────────────────────────
document.getElementById('toggle-json').addEventListener('click', () => {
    const visible = jsonBlock.classList.toggle('visible');
    document.getElementById('toggle-json').innerHTML = visible ? '&#x1F4E5; Hide JSON' : '&#x1F4E6; Show JSON';
});

// ── Copy ───────────────────────────────────────────────────────────
function copyToClipboard(text) {
    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text);
    }
    return new Promise((resolve, reject) => {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try {
            document.execCommand('copy') ? resolve() : reject(new Error('copy failed'));
        } catch (err) {
            reject(err);
        } finally {
            document.body.removeChild(ta);
        }
    });
}

copyBtn.addEventListener('click', () => {
    copyToClipboard(lastJson).then(() => {
        copyBtn.innerHTML = '&#x2705; Copied!';
        setTimeout(() => { copyBtn.innerHTML = '&#x1F4CB; Copy'; }, 1500);
    }).catch(() => {
        renderError('Copy to clipboard failed. Use "Show JSON" and copy the text manually.');
    });
});

// ── Render error ───────────────────────────────────────────────────
function renderError(message) {
    errorMsg.innerHTML = '<div class="error-title">&#x26A0; Error</div><div class="error-detail">' + escapeHtml(message) + '</div>';
    errorMsg.classList.add('visible');
    errorMsg.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
