from __future__ import annotations

import html

from .schemas import Highlight


def locate_phrases(text: str, raw_items: list[dict]) -> list[Highlight]:
    """只接受原文中的精确片段，并计算安全、无重叠的字符位置。"""
    candidates: list[Highlight] = []
    lowered = text.casefold()
    for item in raw_items:
        phrase = str(item.get("text", "")).strip()
        if not phrase:
            continue
        start = lowered.find(phrase.casefold())
        if start < 0:
            continue
        severity = str(item.get("severity", "medium")).lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        candidates.append(
            Highlight(
                text=text[start : start + len(phrase)],
                start=start,
                end=start + len(phrase),
                category=str(item.get("category", "cyberbullying"))[:50],
                severity=severity,
                explanation=str(item.get("explanation", ""))[:300],
            )
        )

    # 优先保留更长片段；拒绝重叠区间，避免产生嵌套、不安全的 HTML。
    selected: list[Highlight] = []
    for candidate in sorted(candidates, key=lambda x: (-(x.end - x.start), x.start)):
        if any(candidate.start < old.end and old.start < candidate.end for old in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda x: x.start)


def render_highlighted_html(text: str, highlights: list[Highlight]) -> str:
    pieces: list[str] = []
    cursor = 0
    for item in highlights:
        pieces.append(html.escape(text[cursor : item.start]))
        title = html.escape(item.explanation, quote=True)
        category = html.escape(item.category, quote=True)
        severity = html.escape(item.severity, quote=True)
        marked = html.escape(text[item.start : item.end])
        pieces.append(
            f'<mark class="cyber-highlight severity-{severity}" '
            f'data-category="{category}" title="{title}">{marked}</mark>'
        )
        cursor = item.end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)
