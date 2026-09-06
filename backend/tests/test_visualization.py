from backend.app.schemas import Highlight
from backend.app.visualization import locate_phrases, render_highlighted_html


def test_locate_rejects_hallucinated_phrase_and_overlap():
    text = "You are a stupid loser."
    items = [
        {"text": "stupid loser", "category": "insult", "severity": "medium"},
        {"text": "loser", "category": "insult", "severity": "medium"},
        {"text": "not in source", "category": "insult", "severity": "low"},
    ]
    result = locate_phrases(text, items)
    assert len(result) == 1
    assert result[0].text == "stupid loser"


def test_html_is_escaped():
    text = '<script>alert(1)</script> stupid'
    highlights = [Highlight(text="stupid", start=26, end=32, category="insult")]
    rendered = render_highlighted_html(text, highlights)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<mark" in rendered
