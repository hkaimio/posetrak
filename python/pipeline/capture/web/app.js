// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

// Operator UI of the capture daemon: device wall, arm/start/stop controls,
// checklist, alerts and the control lease. See api.py for the protocol.

import { html, render, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "./vendor/htm-preact-standalone.js";
import { chosenSetup, formatDuration, formatStorage, initialState, permissions, reduce } from "./store.js";

const LEASE_REFRESH_MS = 10_000;
const RECONNECT_MS = [500, 1000, 2000, 4000];

// -- identity ----------------------------------------------------------------

function remembered(key, make) {
  try {
    let value = localStorage.getItem(key);
    if (!value) {
      value = make();
      localStorage.setItem(key, value);
    }
    return value;
  } catch {
    return make();
  }
}

const clientId = remembered("posetrak.clientId", () => crypto.randomUUID?.() ?? String(Math.random()).slice(2));
const clientLabel = remembered("posetrak.clientLabel", () => `Phone ${Math.floor(Math.random() * 900 + 100)}`);

// The token arrives in the URL fragment (never sent to the server or logged),
// is kept for later visits and removed from the address bar.
function readToken() {
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("token");
  try {
    if (fromHash) {
      localStorage.setItem("posetrak.token", fromHash);
      history.replaceState(null, "", location.pathname + location.search);
      return fromHash;
    }
    return localStorage.getItem("posetrak.token");
  } catch {
    return fromHash;
  }
}

const token = readToken();

// -- server calls ------------------------------------------------------------

async function call(method, path, body) {
  const response = await fetch(`/api${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "X-Client-Id": clientId,
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail : detail?.error ?? response.statusText;
    throw Object.assign(new Error(message), { status: response.status, detail });
  }
  return data;
}

// -- connection --------------------------------------------------------------

function useDaemon() {
  const [state, dispatch] = useReducer(reduce, undefined, initialState);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const snap = await call("GET", "/state");
      dispatch({ type: "snapshot", ...snap });
    } catch (e) {
      setError(e.status === 401 ? "The access token was refused. Scan the QR code again." : e.message);
    }
  }, []);

  useEffect(() => {
    if (!token) {
      setError("No access token. Scan the QR code shown by the daemon.");
      return undefined;
    }
    let socket;
    let attempt = 0;
    let closed = false;
    let timer;

    const open = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/api/events?token=${encodeURIComponent(token)}`);
      socket.onopen = () => {
        attempt = 0;
        setError(null);
        dispatch({ type: "connection", connected: true });
      };
      socket.onmessage = (ev) => dispatch(JSON.parse(ev.data));
      socket.onclose = (ev) => {
        dispatch({ type: "connection", connected: false });
        if (closed) return;
        if (ev.code === 4401) {
          setError("The access token was refused. Scan the QR code again.");
          return;
        }
        timer = setTimeout(open, RECONNECT_MS[Math.min(attempt++, RECONNECT_MS.length - 1)]);
      };
    };
    open();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);

  // Events were lost for this slow client: ask for the complete state.
  useEffect(() => {
    if (state.stale) refresh();
  }, [state.stale, refresh]);

  return { state, dispatch, error, setError, refresh };
}

function useLeaseKeeper(mine) {
  useEffect(() => {
    if (!mine) return undefined;
    const timer = setInterval(() => call("POST", "/lease", { label: clientLabel }).catch(() => {}), LEASE_REFRESH_MS);
    return () => clearInterval(timer);
  }, [mine]);
}

// Keep the screen on while recording so the operator can see problems without touching the phone.
// Browsers grant a wake lock only in a secure context, so over plain http on the
// capture WLAN this quietly does nothing.
function useWakeLock(active) {
  useEffect(() => {
    if (!active || !("wakeLock" in navigator)) return undefined;
    let sentinel;
    const acquire = () => navigator.wakeLock.request("screen").then((s) => (sentinel = s), () => {});
    acquire();
    const onVisible = () => document.visibilityState === "visible" && acquire();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      sentinel?.release().catch(() => {});
    };
  }, [active]);
}

function useRecordingTimer(phase) {
  const [startedAt, setStartedAt] = useState(null);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (phase === "recording") setStartedAt((t) => t ?? Date.now());
    else setStartedAt(null);
  }, [phase]);
  useEffect(() => {
    if (!startedAt) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [startedAt]);
  return startedAt ? (now - startedAt) / 1000 : null;
}

// -- views -------------------------------------------------------------------

const STATE_CLASS = { recording: "rec", ready: "ok", armed: "ok", fault: "bad", offline: "bad", stopped: "", stopping: "warn" };

function CameraTile({ cam, options, override, canControl, onAction, onMode }) {
  const lowBattery = cam.battery_pct != null && cam.battery_pct < 15;
  const lowStorage = cam.storage_remaining_s != null && cam.storage_remaining_s < 300;
  const cls = `tile ${cam.state}${cam.enabled ? "" : " disabled"}`;
  return html`
    <div class=${cls}>
      <div class="name">${cam.label}</div>
      <div class=${`pill ${STATE_CLASS[cam.state] ?? ""}`}>${cam.enabled ? cam.state : "excluded"}</div>
      ${cam.fault && html`<div class="fault-text">${cam.fault}</div>`}
      <div class="meta">
        <span><i class=${`dot ${cam.link === "up" ? "on" : ""}`}></i>link</span><span>${cam.link}</span>
        <span>battery</span><span class=${lowBattery ? "low" : ""}>${cam.battery_pct == null ? "–" : `${cam.battery_pct} %`}</span>
        <span>storage</span><span class=${lowStorage ? "low" : ""}>${formatStorage(cam.storage_remaining_s)}</span>
        <span>control</span><span>${cam.confidence}</span>
        ${cam.enabled && html`<span>mode</span><span>${cam.mode ?? "–"}${cam.settings_control === "manual" ? " (set by hand)" : ""}</span>`}
        ${cam.overheating && html`<span class="low">heat</span><span class="low">hot</span>`}
      </div>
      ${canControl && cam.enabled && options?.modes?.length > 0 &&
      html`
        <select class="mode-select" value=${override ?? ""} onChange=${(e) => onMode(cam.label, e.target.value)}>
          <option value="">mode from setup</option>
          ${options.modes.map((m) => html`<option key=${m} value=${m}>${m}</option>`)}
        </select>`}
      ${canControl &&
      html`
        <div class="actions">
          ${cam.enabled &&
          cam.state === "fault" &&
          html`<button class="small" onClick=${() => onAction("override", cam.label, true)}>It is recording</button>`}
          ${cam.enabled && html`<button class="small" onClick=${() => onAction("manual", cam.label)}>By hand</button>`}
          <button class="small" onClick=${() => onAction("enabled", cam.label, !cam.enabled)}>${cam.enabled ? "Exclude" : "Include"}</button>
        </div>`}
    </div>`;
}

function Alerts({ alerts }) {
  return alerts.map(
    (a) => html`
      <div class="banner bad" key=${`${a.camera}/${a.kind}`}>
        <strong>${a.camera ?? "Capture"} · ${a.kind.replaceAll("_", " ")}</strong>
        ${a.message}
      </div>`,
  );
}

function Report({ report }) {
  if (!report) return null;
  const entries = Object.entries(report.outcomes ?? {});
  return html`
    <h2>${report.title}</h2>
    <div class="wall">
      ${entries.map(
        ([label, o]) => html`
          <div class="tile" key=${label}>
            <div class="name">${label}</div>
            <div class=${`outcome ${o.ok ? "ok" : "bad"}`}>${o.ok ? "✓ ok" : `✗ ${o.error ?? "failed"}`}</div>
            <div class="muted">${o.ok ? o.confidence : ""}</div>
          </div>`,
      )}
    </div>`;
}

function Checklist({ items, canControl, onTick }) {
  if (!items.length) return null;
  return html`
    <h2>Checklist</h2>
    <div class="checklist">
      ${items.map(
        (i) => html`
          <label key=${i.id}>
            <input type="checkbox" checked=${i.done} disabled=${!canControl} onChange=${(e) => onTick(i.id, e.target.checked)} />
            <span>${i.label}${i.camera ? html` <span class="muted">(${i.camera})</span>` : ""}</span>
          </label>`,
      )}
    </div>`;
}

function LeaseBar({ state, perms, onTake }) {
  const { holder, label } = state.lease;
  if (perms.mine) return html`<span class="pill ok">you are in control</span>`;
  if (!holder) return html`<button class="primary small" onClick=${() => onTake(false)}>Take control</button>`;
  return html`
    <span class="pill warn">${label ?? "another device"} is in control</span>
    <button class="small" onClick=${() => confirm(`Take control from ${label ?? "the other device"}?`) && onTake(true)}>Take over</button>`;
}

function App() {
  const { state, dispatch, error, setError, refresh } = useDaemon();
  const perms = permissions(state, clientId);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [setup, setSetup] = useState(null);
  const [overrides, setOverrides] = useState({});
  const elapsed = useRecordingTimer(state.phase);
  useLeaseKeeper(perms.mine);
  useWakeLock(state.phase === "recording");

  const run = useCallback(
    async (fn) => {
      setBusy(true);
      try {
        await fn();
        setError(null);
      } catch (e) {
        setError(e.message);
        if (e.status === 409) refresh();
      } finally {
        setBusy(false);
      }
    },
    [refresh, setError],
  );

  const takeLease = (takeover) => run(() => call("POST", "/lease", { label: clientLabel, takeover }));
  const arm = () =>
    run(async () => {
      const picked = Object.fromEntries(Object.entries(overrides).filter(([, m]) => m));
      const r = await call("POST", "/capture/arm", { setup: chosen, modes: picked });
      if (!r.ready) {
        const problems = Object.entries(r.problems).map(([cam, list]) => `${cam}: ${list.join(", ")}`);
        const open = r.open_checklist.length ? [`checklist open: ${r.open_checklist.join(", ")}`] : [];
        throw new Error(`Not ready. ${[...problems, ...open].join("; ")}`);
      }
    });
  const start = () =>
    run(async () => {
      setConfirming(false);
      const r = await call("POST", "/capture/start", {});
      dispatch({ type: "report", report: { ...r, title: r.ok ? "Started" : "Start incomplete" } });
    });
  const stop = () =>
    run(async () => {
      const r = await call("POST", "/capture/stop");
      dispatch({ type: "report", report: { ...r, title: r.ok ? "Stopped" : "Stop incomplete" } });
    });
  const onAction = (what, label, value) =>
    run(() => {
      const base = `/cameras/${encodeURIComponent(label)}`;
      if (what === "override") return call("POST", `${base}/override`, { recording: value });
      if (what === "enabled") return call("POST", `${base}/enabled`, { enabled: value });
      return call("POST", `${base}/manual`);
    });
  const onTick = (id, done) => run(() => call("POST", `/checklist/${encodeURIComponent(id)}`, { done }));

  const chosen = chosenSetup(state, setup);
  const onMode = (label, mode) => setOverrides((o) => ({ ...o, [label]: mode }));

  const cameras = useMemo(() => Object.values(state.cameras).sort((a, b) => a.label.localeCompare(b.label)), [state.cameras]);
  const recording = state.phase === "recording";

  return html`
    <header class="top">
      <h1>posetrak capture</h1>
      <span class=${`pill ${state.connected ? "ok" : "bad"}`}>${state.connected ? "connected" : "reconnecting…"}</span>
    </header>
    <div class="banner"><div class="row"><${LeaseBar} state=${state} perms=${perms} onTake=${takeLease} /></div></div>
    ${error &&
    html`<div class="banner bad"><div class="row"><span>${error}</span><button class="small" onClick=${() => setError(null)}>OK</button></div></div>`}
    <${Alerts} alerts=${state.alerts} />

    <div class=${`phase ${state.phase}`}>${state.phase}</div>
    ${elapsed != null && html`<div class="timer">${formatDuration(elapsed)}</div>`}

    <div class="controls">
      ${perms.canStart
        ? html`<button class="go" disabled=${busy} onClick=${() => setConfirming(true)}>START ${perms.enabledCount} cameras</button>`
        : recording
          ? html`<button class="stop" disabled=${busy || !perms.canStop} onClick=${stop}>STOP</button>`
          : html`<button class="primary" disabled=${busy || !perms.canArm} onClick=${arm}>${state.phase === "stopped" ? "Arm for next take" : "Arm cameras"}</button>`}
      ${perms.canStart && html`<button disabled=${busy} onClick=${arm}>Re-check cameras</button>`}
    </div>

    ${state.modes.setups.length > 1 &&
    html`
      <label class="setup-pick">setup
        <select value=${chosen ?? ""} disabled=${!perms.canArm} onChange=${(e) => setSetup(e.target.value)}>
          ${state.modes.setups.map((s) => html`<option key=${s} value=${s}>${s}</option>`)}
        </select>
      </label>`}

    <${Report} report=${state.report} />
    <${Checklist} items=${state.checklist} canControl=${perms.mine} onTick=${onTick} />

    <h2>Cameras</h2>
    <div class="wall">
      ${cameras.map((cam) => html`<${CameraTile} key=${cam.label} cam=${cam} options=${state.modes.cameras[cam.label]} override=${overrides[cam.label]} canControl=${perms.mine && !recording} onAction=${onAction} onMode=${onMode} />`)}
    </div>
    ${!cameras.length && html`<p class="muted">No cameras configured.</p>`}

    ${confirming &&
    html`
      <div class="sheet" onClick=${() => setConfirming(false)}>
        <div onClick=${(e) => e.stopPropagation()}>
          <strong>Start recording on ${perms.enabledCount} cameras?</strong>
          <button class="go" onClick=${start}>Start</button>
          <button onClick=${() => setConfirming(false)}>Cancel</button>
        </div>
      </div>`}`;
}

render(html`<${App} />`, document.getElementById("app"));
