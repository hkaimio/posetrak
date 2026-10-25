// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

// Client-side mirror of the controller state.
//
// The daemon sends a complete snapshot when a WebSocket connects and then one
// event per change. `reduce` folds both into a plain object, so the view is a
// pure function of the state and the logic can be tested without a browser.
// Every event carries a sequence number; a gap means the daemon dropped events
// for a slow client, and the caller must fetch a fresh snapshot.

export const initialState = () => ({
  connected: false,
  seq: 0,
  phase: "idle",
  captureId: null,
  cameras: {},
  checklist: [],
  // Setups and capture modes the operator can choose from; `selected` is the setup of the last arm.
  modes: { setups: [], default: null, cameras: {}, selected: null },
  // Copying the recordings off the cameras after a capture: whether it is possible, automatic,
  // running, and the progress of every camera.
  offload: { available: false, auto: false, busy: false, capture_id: null, cameras: {} },
  alerts: [],
  lease: { holder: null, label: null, expires_in_s: null },
  // Outcome of the last start/stop command, per camera, for the confirmation wall.
  report: null,
  // Set when events were lost and a snapshot is needed.
  stale: false,
});

const alertKey = (a) => `${a.camera ?? ""}/${a.kind}`;

export function applySnapshot(state, snap) {
  return {
    ...state,
    seq: snap.seq,
    phase: snap.phase,
    captureId: snap.capture_id,
    cameras: snap.cameras,
    checklist: snap.checklist,
    modes: snap.modes ?? state.modes,
    offload: snap.offload ?? state.offload,
    alerts: snap.alerts,
    lease: snap.lease ?? state.lease,
    stale: false,
  };
}

export function reduce(state, msg) {
  if (msg.type === "snapshot") return applySnapshot(state, msg);
  if (msg.type === "connection") return { ...state, connected: msg.connected };
  if (msg.type === "report") return { ...state, report: msg.report };
  if (msg.type !== "event") return state;

  if (msg.seq <= state.seq) return state; // already reflected in a snapshot
  const next = { ...state, seq: msg.seq, stale: state.stale || msg.seq !== state.seq + 1 };
  const d = msg.data;
  switch (msg.kind) {
    case "phase": {
      const starting = d.phase === "armed" || d.phase === "arming";
      return { ...next, phase: d.phase, captureId: d.capture_id, report: starting ? null : next.report };
    }
    case "camera":
      return { ...next, cameras: { ...next.cameras, [d.label]: d } };
    case "checklist":
      return {
        ...next,
        checklist: next.checklist.map((i) => (i.id === d.item ? { ...i, done: d.done } : i)),
      };
    case "offload":
      return { ...next, offload: d };
    case "plan":
      return { ...next, checklist: d.checklist, modes: { ...next.modes, selected: d.setup } };
    case "alert": {
      const { active, ...alert } = d;
      const rest = next.alerts.filter((a) => alertKey(a) !== alertKey(alert));
      return { ...next, alerts: active ? [...rest, alert] : rest };
    }
    case "lease": {
      const { taken_from, ...lease } = d;
      return { ...next, lease };
    }
    default:
      return next;
  }
}

// What the operator may do now, derived from the phase and the lease.
export function permissions(state, clientId) {
  const mine = state.lease.holder === clientId;
  const enabled = Object.values(state.cameras).filter((c) => c.enabled);
  return {
    mine,
    canArm: mine && ["idle", "stopped", "armed"].includes(state.phase) && !state.offload.busy,
    canStart: mine && state.phase === "armed",
    canStop: mine && state.phase === "recording",
    enabledCount: enabled.length,
  };
}

// The setup the next arm uses: the operator's pick, else the one of the last arm, else the
// daemon's default, else the first (the picker shows the first one when nothing is chosen).
export function chosenSetup(state, picked) {
  const { setups, selected, default: fallback } = state.modes;
  return picked ?? selected ?? fallback ?? setups[0] ?? null;
}

// Progress of copying the recordings, for display: bytes copied of the bytes to copy over all cameras.
export function offloadSummary(offload) {
  const cams = Object.values(offload.cameras);
  const done = cams.reduce((n, c) => n + c.bytes_done, 0);
  const total = cams.reduce((n, c) => n + c.bytes_total, 0);
  const failed = cams.filter((c) => c.state === "failed" || c.state === "skipped").length;
  return { done, total, fraction: total > 0 ? Math.min(1, done / total) : 0, failed, cameras: cams.length };
}

export function formatDuration(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

export function formatStorage(seconds) {
  if (seconds == null) return "–";
  const m = Math.floor(seconds / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
}
