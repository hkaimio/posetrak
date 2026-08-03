# Dependency license analysis
*Personal notes — not for publication. As of 2026-07-01.*

---

## Conclusion up front

**Use AGPL-3.0.** The ultralytics package (AGPL-3.0) is in the mandatory dependency set; it is the binding constraint and effectively forces the combined work to AGPL-3.0 regardless of what license is chosen for the project's own code. All other dependencies are compatible with AGPL-3.0.

---

## Dependency inventory

| Package | License | Notes |
|---|---|---|
| **ultralytics** | **AGPL-3.0** | Binding constraint — see below |
| PySide6 | LGPL-3.0 / GPL-2.0 / GPL-3.0 (choice) | See PySide6 section |
| torch / torchvision | BSD-3-Clause | |
| numpy / scipy / pandas | BSD-3-Clause | |
| opencv-python | Apache-2.0 | |
| timm | Apache-2.0 | |
| onnxruntime-gpu | MIT | |
| hydra-core | MIT | |
| omegaconf | BSD | |
| lap | BSD-2-Clause | |
| matplotlib | PSF-derived (permissive) | |
| h5py / av / pyyaml / click | BSD / MIT | |
| rtmlib | Apache-2.0 (likely — no metadata; verify upstream) | |
| **Cutie** (hkchengrex/Cutie) | **MIT** | Manually cloned, not a pip dep |
| SAM2 (Meta) | Apache-2.0 (code + weights) | See SAM2 section |
| Eigen (C++) | MPL-2.0 | File-level copyleft only |
| Pinocchio (C++) | BSD-2-Clause | |
| Boost (C++) | BSL-1.0 | Very permissive |
| Catch2 (C++ tests) | BSL-1.0 | Test-only |

---

## ultralytics — the binding constraint (AGPL-3.0)

AGPL-3.0 is the most restrictive license in the stack and is in the **mandatory** base dependencies.

**What this means:**
- **GPL-2.0-only is ruled out** — AGPL-3.0 is incompatible with GPL-2.0.
- **GPL-3.0 is technically allowed** but: combining GPL-3.0 code with an AGPL-3.0 dependency means the combined work must be distributed under AGPL-3.0 anyway (the network-use clause of AGPL propagates). There is no practical difference between choosing GPL-3.0 and AGPL-3.0 when the dep is AGPL.
- **Permissive licenses (MIT, Apache, BSD) on own code are allowed** in the sense that AGPL can consume them — but if you release under MIT, downstream users who package your code with ultralytics still face AGPL obligations. This creates confusion with no benefit; just use AGPL.
- **Proprietary / commercial use is blocked** without a separate ultralytics commercial license.

**Alternative if AGPL becomes a problem later:**
- Replace ultralytics person detection with RTMDet via rtmlib (already in the stack; Apache-2.0). Detection quality is slightly lower but acceptable when the SAM2 segmentation path is available.
- Replace SAM2 via ultralytics with Meta's own `sam2` pip package (also Apache-2.0).
- This would remove the only AGPL dependency entirely, opening the door to permissive or commercial licensing.

---

## PySide6 — LGPL-3.0

PySide6 offers three license options: LGPL-3.0, GPL-2.0, or GPL-3.0. LGPL-3.0 is the relevant choice for an open-source project that doesn't want to trigger Qt's commercial license terms.

LGPL-3.0 requires that users can swap the PySide6 library for a modified version. For a Python application this is satisfied automatically — users can replace the venv package without any action on our part.

LGPL-3.0 is fully compatible with AGPL-3.0. No additional obligations beyond what AGPL already requires.

---

## Cutie (hkchengrex/Cutie) — MIT

Cutie is MIT licensed, including its weights. No restrictions on use or distribution.

Note: Cutie cannot currently be installed as a pip/uv package — its `pyproject.toml` (hatchling) fails to build because `cchardet` (a transitive dependency via gradio) does not compile on Python 3.13. It is therefore distributed as a manually cloned repository at a conventional location. This has no legal impact but may matter for reproducibility; document the exact commit hash used if shipping a fixed release.

---

## SAM2 (Meta) — Apache-2.0

Both the SAM2 code and the official model weights are released under Apache-2.0, which permits commercial use. This is more permissive than SAM1 (which some sources remember as restricted — that restriction was lifted or misremembered). Ultralytics bundles SAM2 by downloading Meta's official weights; there is no separate Ultralytics license on top of the weights. The AGPL obligation from using the `ultralytics` package is on the package code, not the weights.

**Verify before acting:** check the current [SAM2 repo](https://github.com/facebookresearch/sam2) LICENSE file if making commercial licensing decisions, since weight licenses can change between releases.

---

## Eigen — MPL-2.0

Mozilla Public License 2.0 is a *file-level* copyleft: modifications to MPL-licensed files must be shared, but the MPL does not infect surrounding GPL/AGPL code. Combining Eigen with AGPL-3.0 code is explicitly permitted. No action required.

---

## Summary: which license choices are available?

| License | Status |
|---|---|
| AGPL-3.0 | ✅ Fully consistent; recommended |
| GPL-3.0 | ✅ Technically compatible; combined work is AGPL-3.0 anyway |
| GPL-2.0 only | ❌ Incompatible with AGPL-3.0 (ultralytics) |
| Apache-2.0 / MIT / BSD | ⚠ Allowed for own code, but creates confusion for downstream users who must still follow AGPL |
| Proprietary / commercial | ❌ Requires separate ultralytics commercial license |
