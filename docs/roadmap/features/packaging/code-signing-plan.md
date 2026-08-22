# Windows code signing — implementation plan

> **Status (2026-08-23)**: Proposal only, nothing implemented, and
> **deliberately deferred** — Harri: this is a hobby project with a
> hobby budget, so paying for a certificate/signing service isn't
> worth doing until there's real evidence of user interest in Posetrak
> beyond its current use. The plan below is worth keeping ready for
> that point rather than reconsidering from scratch, but isn't part of
> the near-term installer prototype (see
> [installer-prototype-plan.md](installer-prototype-plan.md)), which
> ships unsigned and documents the resulting SmartScreen warning
> instead.

## Why this matters at all

An unsigned `.exe` downloaded from the internet gets flagged by Windows
SmartScreen with a full-screen "Windows protected your PC" warning —
blue shield icon, no publisher name, a buried "More info → Run anyway"
link most users won't find on their own. For a first release trying to
look like a real, trustworthy tool rather than a random download, this
is a serious first impression problem independent of whether the
software is actually fine.

Signing the installer removes the "unknown publisher" state and shows a
real name instead. It does **not**, by itself, guarantee no warning at
all — see "SmartScreen reputation" below, which is the part most likely
to surprise anyone whose mental model is "signed = trusted immediately."

## What changed recently (the part worth double-checking your assumptions on)

If your reference point is a few years old, the biggest change: **since
June 2023, the CA/Browser Forum requires every code-signing
certificate's private key — not just EV — to live on a hardware token
(a physical USB HSM) or a cloud HSM** (Azure Key Vault, or a signing
service that manages this for you). Downloading a `.pfx` file and
signing locally the way it used to work for a basic ("OV") certificate
is no longer how any CA issues new certificates. This materially changes
the CI story: a physical USB token doesn't plug into an ephemeral
GitHub-hosted runner, which pushes toward either a self-hosted runner or
a cloud-HSM-backed option.

## Certificate options

| Option | Identity check | Key storage | SmartScreen reputation | Rough cost |
|---|---|---|---|---|
| Self-signed | None | Anywhere | None — not trusted by any real user's machine | Free |
| Standard (OV) cert, traditional CA | Individual or business identity | Hardware token or cloud HSM (mandatory since 2023) | Starts at zero; builds up over time as more people run the exact signed binary without it being reported | ~$70-250/yr certificate, plus HSM/token cost |
| EV cert, traditional CA | Stricter business identity vetting | Hardware token or cloud HSM | Immediate — EV has historically skipped the reputation-building wait | ~$300-700+/yr |
| Microsoft Trusted Signing (Azure) | Individual or business identity via Microsoft Entra ID | Managed by Microsoft, no token to handle | Believed favorable given first-party trust chain, not independently confirmed here — verify empirically (see Phase 1) | ~$10/month advertised, no separate cert purchase |

Self-signed is listed only to close it off as an option: it has no
practical value for public distribution, since Windows doesn't trust a
self-signed cert unless each user's machine has it manually added to
their trusted root store.

**Recommendation for the prototype phase: Microsoft Trusted Signing.**
It's the only option here with no physical token to lose or manage, the
cheapest by a wide margin, and integrates with GitHub Actions via an
official signing action using short-lived, federated credentials
(no long-lived secret to store). The tradeoffs (exact SmartScreen
reputation behavior, current eligibility rules for an individual vs.
requiring a registered business) aren't things this doc can state as
settled fact — they need confirming directly against Microsoft's current
program terms before committing, which is exactly what Phase 1 below is
for. Cost figures and requirements throughout this doc are order-of-
magnitude and should be re-verified at that point too — this is a fast-
moving area and prices/programs change.

## SmartScreen reputation — the part a prototype phase actually tests

Signing alone doesn't erase the warning immediately for a non-EV,
non-Trusted-Signing certificate — Microsoft's SmartScreen reputation
system needs to observe enough unique downloads/installs of that
specific signed release without malware reports before it stops
flagging it, which can take days to weeks of real-world usage. This is
normal, not a sign the signing setup is broken, but it's exactly the
kind of thing that's much better to discover with a handful of informed
testers than with a public release announcement.

## Signing mechanics (once a certificate/service is chosen)

Regardless of which option above is chosen, the actual signing step
looks similar: `signtool sign /fd SHA256 /tr <RFC3161 timestamp server>
/td SHA256 <installer>.exe` (or the equivalent wrapper CLI a signing
service provides), run as a step in the release CI workflow after the
Inno Setup installer is compiled. Timestamping (`/tr`/`/td`) matters —
it keeps the signature valid after the certificate itself eventually
expires, by proving the signing happened while the cert was still
valid.

## Phased plan

### Phase 0 — Decide identity and confirm current terms

- Decide personal vs. registered-business identity for whichever
  option is chosen — affects both eligibility and vetting time.
- Confirm Microsoft Trusted Signing's current eligibility rules and
  pricing directly (this doc's numbers are order-of-magnitude, not
  quoted from a current source). If it turns out not to fit (e.g.
  requires a business entity that doesn't exist yet), fall back to a
  standard OV cert from a traditional CA as the prototype option
  instead — the rest of this plan doesn't otherwise depend on which one.

### Phase 1 — Prototype the signing pipeline itself

- Get *a* working certificate/service set up (identity verification is
  the slow part here — start this early, independent of the rest of
  packaging work).
- Wire signing into the release workflow from
  `packaging-design.md` as a single new CI step: sign the compiled
  Inno Setup installer before it's uploaded to the GitHub Release.
- Verify the signature is actually recognized correctly (`signtool
  verify`, and manually checking the installer's Properties → Digital
  Signatures tab in Windows Explorer) — confirming the pipeline works
  mechanically, before worrying about SmartScreen behavior at all.

### Phase 2 — Small group test

- Distribute a signed prototype build to a handful of people who know
  they're testing an early build (not a public release) — ideally on
  machines that have never seen any Posetrak build before, so
  SmartScreen's behavior is observed fresh rather than influenced by a
  tester's own machine already having run an earlier unsigned version.
- Have them record, honestly, what they actually saw on first run:
  no warning at all, a signed-but-still-flagged warning, or the full
  unknown-publisher warning — and roughly how many of them saw each.
  This is the concrete signal for whether the chosen certificate type's
  reputation behavior matches what Phase 0 assumed.
- Re-run this after a second signed release (a version bump) if the
  first round showed warnings, to see whether reputation is actually
  accumulating over time as expected.

### Phase 3 — Decide the real investment based on what Phase 2 showed

- If the prototype option (Trusted Signing or a cheap OV cert) already
  shows clean installs with no warnings for testers: keep it, done.
- If warnings persist longer than seems reasonable: this is the point
  to weigh upgrading to an EV certificate (immediate reputation,
  significantly higher cost) against just accepting a reputation-
  building window for the real public release too, now backed by actual
  observed data instead of a guess.

### Phase 4 — Roll into the general release

- Whatever was validated in Phase 2/3 becomes the standing CI signing
  step for every tagged release, not just the prototype.
- Document the actual first-run experience users should expect
  (`docs/setup.md` or wherever the release install instructions end up
  living, per `packaging-design.md`) — including "if you see a
  SmartScreen warning, here's why and what to click" if reputation
  turns out not to be fully clean by general release.

## Open questions (not resolved here)

1. **Individual vs. business identity** — affects eligibility for every
   option in the table above; not decided here.
2. **Exact current Trusted Signing pricing/eligibility** — this doc's
   numbers are approximate and explicitly need reverifying in Phase 0,
   not treated as settled.
3. **Whether Linux artifacts need anything equivalent** — AppImages
   don't have a SmartScreen-equivalent gate on most distros; this plan
   is Windows-specific and deliberately doesn't address whether Linux
   needs its own trust story (e.g. GPG-signing the AppImage) — worth a
   separate, much shorter look once this Windows plan is underway.
