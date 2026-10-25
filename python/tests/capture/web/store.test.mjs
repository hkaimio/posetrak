// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { formatDuration, formatStorage, initialState, permissions, reduce } from "../../../pipeline/capture/web/store.js";

const cam = (label, extra = {}) => ({ label, state: "ready", enabled: true, ...extra });
const snapshot = (extra = {}) => ({
  type: "snapshot",
  seq: 10,
  phase: "idle",
  capture_id: null,
  cameras: { a: cam("a"), b: cam("b") },
  checklist: [{ id: "mode", label: "Mode set", camera: null, done: false }],
  alerts: [],
  lease: { holder: null, label: null, expires_in_s: null },
  ...extra,
});
const event = (seq, kind, data) => ({ type: "event", seq, kind, ts: "t", data });

test("snapshot replaces state and clears staleness", () => {
  const s = reduce({ ...initialState(), stale: true }, snapshot());
  assert.equal(s.seq, 10);
  assert.equal(s.stale, false);
  assert.deepEqual(Object.keys(s.cameras), ["a", "b"]);
});

test("events already covered by the snapshot are ignored", () => {
  const s = reduce(reduce(initialState(), snapshot()), event(9, "phase", { phase: "armed", capture_id: null }));
  assert.equal(s.phase, "idle");
});

test("events apply in order", () => {
  let s = reduce(initialState(), snapshot());
  s = reduce(s, event(11, "phase", { phase: "armed", capture_id: null }));
  s = reduce(s, event(12, "camera", cam("a", { state: "recording" })));
  s = reduce(s, event(13, "checklist", { item: "mode", done: true }));
  assert.equal(s.phase, "armed");
  assert.equal(s.cameras.a.state, "recording");
  assert.equal(s.checklist[0].done, true);
  assert.equal(s.stale, false);
  assert.equal(s.seq, 13);
});

test("a sequence gap marks the state stale", () => {
  const s = reduce(reduce(initialState(), snapshot()), event(15, "phase", { phase: "armed", capture_id: null }));
  assert.equal(s.stale, true);
});

test("alerts are raised and cleared per camera and kind", () => {
  let s = reduce(initialState(), snapshot());
  const alert = { kind: "link_lost", camera: "a", message: "gone", ts: "t" };
  s = reduce(s, event(11, "alert", { active: true, ...alert }));
  s = reduce(s, event(12, "alert", { active: true, ...alert, kind: "battery_low" }));
  assert.equal(s.alerts.length, 2);
  s = reduce(s, event(13, "alert", { active: false, ...alert }));
  assert.deepEqual(s.alerts.map((a) => a.kind), ["battery_low"]);
});

test("a new capture drops the previous report", () => {
  let s = reduce(initialState(), snapshot());
  s = reduce(s, { type: "report", report: { title: "Stopped", outcomes: {} } });
  s = reduce(s, event(11, "phase", { phase: "arming", capture_id: null }));
  assert.equal(s.report, null);
});

test("lease events update the holder", () => {
  const s = reduce(
    reduce(initialState(), snapshot()),
    event(11, "lease", { holder: "me", label: "Phone", expires_in_s: 30, taken_from: null }),
  );
  assert.equal(s.lease.holder, "me");
  assert.equal("taken_from" in s.lease, false);
});

test("permissions follow the phase and the lease", () => {
  const base = reduce(initialState(), snapshot({ lease: { holder: "me", label: "p", expires_in_s: 30 } }));
  assert.deepEqual(
    [permissions(base, "me").canArm, permissions(base, "me").canStart, permissions(base, "other").canArm],
    [true, false, false],
  );
  const armed = reduce(base, event(11, "phase", { phase: "armed", capture_id: null }));
  assert.equal(permissions(armed, "me").canStart, true);
  const rec = reduce(armed, event(12, "phase", { phase: "recording", capture_id: "c" }));
  assert.equal(permissions(rec, "me").canStop, true);
  assert.equal(permissions(rec, "me").canArm, false);
});

test("formatting", () => {
  assert.equal(formatDuration(75.9), "01:15");
  assert.equal(formatStorage(null), "–");
  assert.equal(formatStorage(300), "5 min");
  assert.equal(formatStorage(7500), "2 h 5 min");
});

test("the snapshot carries the setups and modes, and a plan event updates the checklist and the setup", () => {
  const modes = { setups: ["a", "b"], default: "a", selected: null, cameras: { a: { modes: ["4K"] } } };
  let s = reduce(initialState(), snapshot({ modes }));
  assert.deepEqual(s.modes.setups, ["a", "b"]);
  s = reduce(s, event(11, "plan", { setup: "b", checklist: [{ id: "mode:a", label: "set a", camera: "a", done: false }] }));
  assert.equal(s.modes.selected, "b");
  assert.deepEqual(s.modes.cameras, modes.cameras);
  assert.deepEqual(s.checklist.map((i) => i.id), ["mode:a"]);
});

test("a snapshot without modes keeps the ones known", () => {
  const withModes = reduce(initialState(), snapshot({ modes: { setups: ["a"], default: "a", selected: "a", cameras: {} } }));
  assert.deepEqual(reduce(withModes, snapshot()).modes.setups, ["a"]);
});
