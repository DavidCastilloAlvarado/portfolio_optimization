window.runProgressJob = async function (endpoint, formData, title, initialState) {
    const panel = document.getElementById('job-progress');
    const heading = document.getElementById('job-progress-title');
    const state = document.getElementById('job-progress-state');
    const elapsed = document.getElementById('job-progress-elapsed');
    const track = document.getElementById('job-progress-track');
    const eventList = document.getElementById('job-progress-events');
    let lastSequence = -1;

    function setStatus(status) {
        panel.dataset.status = status;
        panel.setAttribute('aria-busy', String(status === 'running'));
        if (status === 'running') {
            track.removeAttribute('aria-valuenow');
        } else {
            track.setAttribute('aria-valuenow', status === 'completed' ? '100' : '0');
        }
    }

    function appendEvents(events) {
        for (const event of events || []) {
            if (event.sequence <= lastSequence) continue;
            lastSequence = event.sequence;
            const item = document.createElement('li');
            const time = document.createElement('time');
            const tag = document.createElement('span');
            const message = document.createElement('span');
            const date = new Date(event.timestamp);
            time.textContent = Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
            tag.className = 'job-progress-event-tag';
            tag.textContent = `[${event.tag}]`;
            message.textContent = event.message;
            item.append(time, tag, message);
            eventList.appendChild(item);
            while (eventList.children.length > 80) eventList.firstElementChild.remove();
        }
        eventList.scrollTop = eventList.scrollHeight;
    }

    function setElapsed(seconds) {
        const total = Math.max(0, Math.floor(Number(seconds) || 0));
        const minutes = Math.floor(total / 60);
        const remainder = total % 60;
        elapsed.textContent = minutes ? `Elapsed ${minutes}m ${String(remainder).padStart(2, '0')}s` : `Elapsed ${total}s`;
    }

    function finish(status, message) {
        setStatus(status);
        state.textContent = message;
    }

    panel.hidden = false;
    panel.dataset.operation = endpoint.includes('/select/') ? 'selection' : 'optimization';
    heading.textContent = title;
    state.textContent = initialState;
    elapsed.textContent = 'Elapsed 0s';
    eventList.replaceChildren();
    setStatus('running');
    formData.set('progress', 'on');

    try {
        const response = await fetch(endpoint, { method: 'POST', body: formData });
        let payload;
        try {
            payload = await response.json();
        } catch (error) {
            throw new Error(`Server returned an invalid response (HTTP ${response.status}).`);
        }
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);

        if (!payload.job_id) {
            finish('completed', 'Completed successfully');
            return payload;
        }

        while (true) {
            let pollResponse;
            try {
                pollResponse = await fetch(`/jobs/${encodeURIComponent(payload.job_id)}?after=${lastSequence}`, { cache: 'no-store' });
            } catch (error) {
                state.textContent = 'Connection interrupted; reconnecting…';
                await new Promise(resolve => setTimeout(resolve, 1500));
                continue;
            }
            if (pollResponse.status === 404) throw new Error('Progress job expired or is no longer available.');
            if (!pollResponse.ok) {
                state.textContent = 'Unable to read progress; retrying…';
                await new Promise(resolve => setTimeout(resolve, 1500));
                continue;
            }

            const snapshot = await pollResponse.json();
            appendEvents(snapshot.events);
            setElapsed(snapshot.elapsed_seconds);
            if (snapshot.status === 'completed') {
                finish('completed', snapshot.state || 'Completed successfully');
                return snapshot.result;
            }
            if (snapshot.status === 'failed') {
                throw new Error(snapshot.error || 'Analysis failed.');
            }
            state.textContent = snapshot.state || 'Running…';
            await new Promise(resolve => setTimeout(resolve, 700));
        }
    } catch (error) {
        finish('failed', error.message || 'Analysis failed.');
        throw error;
    }
};
