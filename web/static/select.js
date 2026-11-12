const themeToggle = document.getElementById('theme-toggle');
function applyTheme(name) {
    document.documentElement.setAttribute('data-theme', name);
    themeToggle.innerHTML = name === 'dark' ? '&#x1F319; Dark' : '&#x2600;&#xFE0E; Light';
    themeToggle.setAttribute('aria-pressed', String(name === 'dark'));
}
applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');
themeToggle.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('po-theme', next); } catch (e) {}
});

const statusDiv = document.getElementById('stage-status');
const errorMsg = document.getElementById('error-msg');
const results = document.getElementById('results');
const topNSel = document.getElementById('selection_top_n');
const recursionLimitInput = document.getElementById('selection_recursion_limit');
const verdictConcurrencyInput = document.getElementById('selection_verdict_concurrency');
const forceRefresh = document.getElementById('selection_force_refresh');

let lastData = null;
let lastTickers = [];

function escapeHtml(text) {
    return String(text ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function logStage(line, cls) {
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = line;
    statusDiv.appendChild(div);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

function showError(message) {
    errorMsg.textContent = message;
    errorMsg.classList.add('visible');
    errorMsg.focus({ preventScroll: true });
}

function clearError() {
    errorMsg.classList.remove('visible');
    errorMsg.textContent = '';
}

async function runStage(stage, label) {
    clearError();
    statusDiv.replaceChildren();
    logStage(`started: ${label}`);
    results.setAttribute('aria-busy', 'true');
    for (const b of document.querySelectorAll('.sel-controls .btn')) b.disabled = true;
    try {
        const form = new URLSearchParams();
        form.set('stage', stage);
        form.set('selection_top_n', topNSel.value);
        form.set('selection_recursion_limit', recursionLimitInput.value);
        form.set('selection_verdict_concurrency', verdictConcurrencyInput.value);
        form.set('force_refresh', forceRefresh.checked ? 'on' : 'off');
        const data = await window.runProgressJob('/select/run', form, 'Selection · ' + stage, label);
        lastData = data;
        render(data);
        const degraded = Object.keys(data.degraded || {});
        if (degraded.length) {
            for (const k of degraded) logStage(`warn: ${k} degraded — ${data.degraded[k]}`.slice(0, 160), 'warn');
        }
        logStage(`ok: ${stage} complete` + (data.from_cache ? ' (from cache)' : ''), 'ok');
    } catch (e) {
        showError(e.message);
        logStage('error: ' + e.message, 'bad');
    } finally {
        results.setAttribute('aria-busy', 'false');
        for (const b of document.querySelectorAll('.sel-controls .btn')) b.disabled = false;
    }
}

// ── Rendering ────────────────────────────────────────────────────────

function render(data) {
    renderScreen(data);
    if (data.briefing) renderBriefing(data.briefing, data.as_of);
    renderSelection(data);
}

function fmtPct(v) {
    if (v === null || v === undefined) return '&mdash;';
    const p = (v * 100).toFixed(1);
    return `<span class="${v >= 0 ? 'pos' : 'neg'}">${p > 0 ? '+' : ''}${p}%</span>`;
}

function fmtScore(v) {
    return v === null || v === undefined ? '&mdash;' : Number(v).toFixed(3);
}

function renderScreen(data) {
    const s = data.screen || {};
    document.getElementById('briefing-asof').textContent = '';
    document.getElementById('sectors-section').hidden = false;

    const body = document.getElementById('sectors-body');
    body.innerHTML = '';
    const sectors = (data.sectors && data.sectors.length) ? data.sectors : s.sectors;
    for (const sec of sectors || []) {
        const tr = document.createElement('tr');
        const llm = sec.llm;
        const hot = sec.rank !== undefined && sec.rank <= 3 ? ' class="flag"' : '';
        tr.innerHTML = `
            <td class="num"${hot}>${sec.rank !== undefined ? sec.rank : ''}</td>
            <td>${sec.rank !== undefined && sec.rank <= 3 ? '<span class="flag">&#x25B8;</span> ' : ''}${escapeHtml(sec.sector)}</td>
            <td class="num">${sec.n_members}</td>
            <td class="num">${fmtScore(sec.quant_score)}</td>
            <td class="num">${fmtScore(sec.momentum)}</td>
            <td class="num">${fmtScore(sec.breadth)}</td>
            <td>${llm ? `<span class="sel-chip ${escapeHtml(llm.regime)}">${escapeHtml(llm.regime)}</span>` : '<span class="dim">&mdash;</span>'}</td>
            <td>${llm ? escapeHtml(llm.driver) : '<span class="dim">&mdash;</span>'}</td>
            <td class="num">${llm && Number.isFinite(Number(llm.conviction)) ? Number(llm.conviction).toFixed(2) : '&mdash;'}</td>
            <td class="num">${fmtScore(sec.final_score)}</td>
        `;
        body.appendChild(tr);
    }
    if (!s.skipped || s.skipped.length === 0) return;
    logStage(`note: ${s.skipped.length} universe tickers skipped: ${s.skipped.join(', ')}`, 'warn');
}

function renderBriefing(b, asof) {
    document.getElementById('briefing-section').hidden = false;
    document.getElementById('briefing-asof').textContent = asof || '';
    document.getElementById('briefing-summary').textContent = b.regime_summary || '';
    const risk = Number(b.overall_market_risk);
    const safeRisk = Number.isFinite(risk) ? risk : 0;
    const statsDiv = document.getElementById('briefing-stats');
    statsDiv.innerHTML = '';
    const cards = [
        { label: 'Overall market risk', value: safeRisk.toFixed(2), cls: safeRisk > 0.6 ? 'negative' : 'positive', sub: '1 = strongly risk-off' },
        { label: 'Sectors assessed', value: String((dataSectors(b)) || ''), cls: 'accent', sub: 'from the LLM briefing' },
    ];
    for (const c of cards) {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `<div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div><div class="sub">${c.sub}</div>`;
        statsDiv.appendChild(card);
    }
    const risks = document.getElementById('briefing-risks');
    risks.innerHTML = (b.macro_risks || []).map(r => `<li>${escapeHtml(r)}</li>`).join('') || '<li>&mdash;</li>';
    const themes = document.getElementById('briefing-themes');
    themes.innerHTML = (b.trending_themes || []).map(t => `<li>${escapeHtml(t)}</li>`).join('') || '<li>&mdash;</li>';
}

function dataSectors(b) {
    // briefing payload only carries scalars; sector rows render in the table
    return lastData && lastData.sectors ? lastData.sectors.length : null;
}

function renderSelection(data) {
    const sel = data.selection || [];
    if (!sel.length) return;
    document.getElementById('selection-section').hidden = false;
    document.getElementById('selection-asof').textContent = data.as_of || '';
    const body = document.getElementById('selection-body');
    body.innerHTML = '';
    lastTickers = sel.map(n => n.ticker);
    for (let i = 0; i < sel.length; i++) {
        const n = sel[i];
        const m = n.metrics || {};
        const v = n.verdict;
        const tr = document.createElement('tr');
        const overhangs = (v && v.overhangs) || [];
        const flags = [];
        if (v && v.event_risk) flags.push(`earnings in ${v.earnings_in_days}d`);
        if (!v) flags.push('no LLM verdict');
        const pe = n.trailing_pe ? Number(n.trailing_pe).toFixed(1) : '&mdash;';
        tr.innerHTML = `
            <td class="num">${i + 1}</td>
            <td><strong>${escapeHtml(n.ticker)}</strong><div class="dim">${escapeHtml(n.name)}</div></td>
            <td>${escapeHtml(n.sector)}</td>
            <td class="num">${fmtScore(n.finance_score)}</td>
            <td class="num">${n.conviction !== null && n.conviction !== undefined && Number.isFinite(Number(n.conviction)) ? Number(n.conviction).toFixed(2) : '&mdash;'}</td>
            <td class="num">${fmtScore(n.final_score)}</td>
            <td class="num">${fmtPct(m.ret_1m)}</td>
            <td class="num">${fmtPct(m.ret_3m)}</td>
            <td class="num">${fmtPct(m.ret_6m)}</td>
            <td class="num">${fmtPct(m.off_high_52w)}</td>
            <td class="num">${pe}</td>
            <td>${v ? escapeHtml(v.thesis) : '<span class="dim">&mdash;</span>'}</td>
            <td>${[...overhangs.map(escapeHtml), ...flags.map(f => `<span class="flag">&#x26A0; ${escapeHtml(f)}</span>`)].join('<br>')}</td>
        `;
        body.appendChild(tr);
    }
    const vetoed = data.vetoed || [];
    const box = document.getElementById('vetoed-box');
    if (vetoed.length) {
        box.hidden = false;
        document.getElementById('vetoed-chips').innerHTML =
            vetoed.map(t => `<span class="sel-chip" style="margin:2px">${escapeHtml(t)}</span>`).join(' ');
    } else {
        box.hidden = true;
    }
}

document.getElementById('btn-send-optimizer').addEventListener('click', () => {
    if (!lastTickers.length) return;
    location.href = '/?shares=' + encodeURIComponent(lastTickers.join(','));
});

document.getElementById('btn-stage-screen').addEventListener('click', () => runStage('screen', 'Screening the S&P 100 (prices + quotes)...'));
document.getElementById('btn-stage-briefing').addEventListener('click', () => runStage('briefing', 'Market briefing agent researching macro, geopolitics and sectors (10-20 min)...'));
document.getElementById('btn-stage-verdicts').addEventListener('click', () => runStage('verdicts', 'Name verdict agents researching each candidate (a few minutes each)...'));
document.getElementById('btn-stage-all').addEventListener('click', () => runStage('all', 'Running the full funnel (quant + market briefing + name verdicts, 20-40 min first run)...'));
