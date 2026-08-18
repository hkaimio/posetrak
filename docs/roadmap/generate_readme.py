#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Generate docs/roadmap/README.md from each feature's status.md frontmatter.

Each feature lives in docs/roadmap/features/<slug>/ and must have a status.md
whose first lines are a TOML frontmatter block, fenced with ```toml (NOT bare
`+++` delimiters -- GitHub has no special handling for TOML frontmatter, so a
bare `+++` block just renders as an ordinary paragraph and visually collapses
onto one line; a fenced code block renders correctly on every Markdown host):

    ```toml
    name = "Human-readable feature name"
    status = "proposal"          # proposal | in_progress | complete | released
    progress_pct = 42            # optional int, only meaningful for in_progress
    description = "One or two sentences shown on the roadmap page."
    categories = ["tracker-core", "ukf-tuning"]
    target_release = "v1.0"      # or "TBD" if not yet scheduled
    last_updated = 2026-08-06    # bare TOML date, no quotes
    ```

    # Rest of the document is free-form status prose (implementation status,
    # known issues, etc).

Run this whenever a feature's status.md changes -- it is idempotent, re-running
with no status.md changes produces byte-identical output:

    python docs/roadmap/generate_readme.py

A feature folder with no status.md, or one whose frontmatter fails to parse or
is missing a required field, is skipped with a warning on stderr rather than
failing the whole run -- so one broken doc doesn't block the rest of the
roadmap page from regenerating.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys
import tomllib
from pathlib import Path

ROADMAP_DIR = Path(__file__).resolve().parent
FEATURES_DIR = ROADMAP_DIR / "features"
OUTPUT_PATH = ROADMAP_DIR / "README.md"

DOC_SUFFIXES = {".md", ".txt"}

STATUS_LABEL = {
    "proposal": "\U0001f4dd Proposal",
    "in_progress": "\U0001f6a7 In progress",
    "complete": "✅ Complete",
    "released": "\U0001f680 Released",
}
# Display order within a category group -- most-active work first.
STATUS_ORDER = {"in_progress": 0, "proposal": 1, "complete": 2, "released": 3}
REQUIRED_FIELDS = (
    "name",
    "status",
    "description",
    "categories",
    "target_release",
    "last_updated",
)
UNSCHEDULED_LABELS = {"tbd", "unscheduled", "backlog", ""}


@dataclasses.dataclass
class Feature:
    slug: str
    name: str
    status: str
    description: str
    categories: list[str]
    target_release: str
    last_updated: datetime.date | str
    progress_pct: int | None
    docs: list[Path]

    @property
    def status_label(self) -> str:
        label = STATUS_LABEL[self.status]
        if self.status == "in_progress" and self.progress_pct is not None:
            return f"{label} ({self.progress_pct}%)"
        return label

    @property
    def primary_category(self) -> str:
        return self.categories[0] if self.categories else "uncategorized"


FRONTMATTER_FENCE = "```toml"


def _extract_frontmatter(text: str, source: Path) -> str:
    if not text.startswith(FRONTMATTER_FENCE):
        raise ValueError(
            f"{source}: missing leading '{FRONTMATTER_FENCE}' frontmatter fence"
        )
    body = text[len(FRONTMATTER_FENCE) :]
    end = body.find("\n```")
    if end == -1:
        raise ValueError(f"{source}: unterminated frontmatter (no closing '```' line)")
    return body[:end]


def load_feature(folder: Path) -> Feature | None:
    status_file = folder / "status.md"
    if not status_file.exists():
        print(
            f"warning: {folder.name}/ has no status.md -- skipped "
            f"(see the docstring at the top of {Path(__file__).name} for the required format)",
            file=sys.stderr,
        )
        return None

    text = status_file.read_text(encoding="utf-8")
    try:
        meta = tomllib.loads(_extract_frontmatter(text, status_file))
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"warning: {status_file}: {exc} -- skipped", file=sys.stderr)
        return None

    missing = [field for field in REQUIRED_FIELDS if field not in meta]
    if missing:
        print(
            f"warning: {status_file}: frontmatter missing required field(s) {missing} -- skipped",
            file=sys.stderr,
        )
        return None

    if meta["status"] not in STATUS_LABEL:
        print(
            f"warning: {status_file}: unknown status {meta['status']!r} "
            f"(expected one of {sorted(STATUS_LABEL)}) -- skipped",
            file=sys.stderr,
        )
        return None

    docs = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in DOC_SUFFIXES),
        key=lambda p: (p.name != "status.md", p.name.lower()),
    )

    return Feature(
        slug=folder.name,
        name=str(meta["name"]),
        status=meta["status"],
        description=str(meta["description"]).strip(),
        categories=list(meta["categories"]),
        target_release=str(meta["target_release"]),
        last_updated=meta["last_updated"],
        progress_pct=meta.get("progress_pct"),
        docs=docs,
    )


def collect_features() -> list[Feature]:
    features = []
    for folder in sorted(FEATURES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        feature = load_feature(folder)
        if feature is not None:
            features.append(feature)
    return features


def _release_sort_key(release: str) -> tuple[int, str]:
    if release.strip().lower() in UNSCHEDULED_LABELS:
        return (1, release)
    return (0, release)


def _render_feature(feature: Feature) -> list[str]:
    lines = [f"#### {feature.name}", ""]
    tags = ", ".join(f"`{c}`" for c in feature.categories)
    # One line, joined with middot separators rather than markdown hard line
    # breaks (trailing double-spaces) -- the project's pre-commit hook strips
    # trailing whitespace, which would silently collapse those onto one line
    # anyway on the next commit.
    lines.append(
        f"**Status:** {feature.status_label} · **Categories:** {tags} · "
        f"**Last updated:** {feature.last_updated}"
    )
    lines.append("")
    lines.append(feature.description)
    lines.append("")
    lines.append("Documents:")
    for doc in feature.docs:
        lines.append(f"- [{doc.name}](features/{feature.slug}/{doc.name})")
    lines.append("")
    return lines


def render_readme(features: list[Feature]) -> str:
    lines = [
        "# Posetrak Roadmap",
        "",
        "Auto-generated by [`generate_readme.py`](generate_readme.py) from each feature's "
        "`status.md` frontmatter.",
        "**Do not edit this file directly** -- edit the relevant `status.md` and re-run:",
        "",
        "```",
        "python docs/roadmap/generate_readme.py",
        "```",
        "",
    ]

    by_release: dict[str, list[Feature]] = {}
    for feature in features:
        by_release.setdefault(feature.target_release, []).append(feature)

    for release in sorted(by_release, key=_release_sort_key):
        label = (
            "Unscheduled"
            if release.strip().lower() in UNSCHEDULED_LABELS
            else release
        )
        lines.append(f"## {label}")
        lines.append("")

        by_category: dict[str, list[Feature]] = {}
        for feature in by_release[release]:
            by_category.setdefault(feature.primary_category, []).append(feature)

        show_category_headers = len(by_category) > 1
        for category in sorted(by_category):
            if show_category_headers:
                lines.append(f"### {category}")
                lines.append("")
            ordered = sorted(
                by_category[category],
                key=lambda f: (STATUS_ORDER.get(f.status, 99), f.name),
            )
            for feature in ordered:
                lines.extend(_render_feature(feature))

    lines.append("---")
    lines.append("")
    lines.append("## Other planning documents")
    lines.append("")
    lines.append(
        "- [first-release-backlog.md](first-release-backlog.md) -- prioritized backlog items "
        "not (yet) written up as their own feature"
    )
    lines.append(
        "- [features/tracking-crisis-debugging-log.md](features/tracking-crisis-debugging-log.md)"
        " -- working notes for the multi-session UKF tracking-crisis investigation that "
        "motivated several of the tracker-core features above"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    features = collect_features()
    if not features:
        print("error: no features with a valid status.md found", file=sys.stderr)
        return 1

    OUTPUT_PATH.write_text(render_readme(features), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROADMAP_DIR.parent.parent)} ({len(features)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
