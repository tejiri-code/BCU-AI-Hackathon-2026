from apexmind import scoring


def test_evidence_quality_counts():
    items = [
        {"title": "A", "snippet": "x" * 100, "source": "wikipedia"},
        {"title": "B", "snippet": "y" * 50, "source": "web"},
    ]
    eq = scoring.evidence_quality(items)
    assert eq["n_docs"] == 2 and eq["n_wikipedia"] == 1 and eq["has_wikipedia"]
    assert eq["total_chars"] == 150


def test_support_score_overlap():
    options = {"A": "foreign internal defense reconnaissance", "B": "cyber warfare"}
    ev = "the unit conducts foreign internal defense and special reconnaissance"
    assert scoring.support_score("A", options, ev) > 0.5
    assert scoring.support_score("B", options, ev) == 0.0


def test_confidence_agreement_boosts():
    items = [{"title": "T", "snippet": "z" * 80, "source": "wikipedia"}]
    options = {"A": "alpha", "B": "beta"}
    rag = {"answer": "A", "parsed_from": "pattern"}
    cb_agree = {"answer": "A", "parsed_from": "pattern"}
    cb_disagree = {"answer": "B", "parsed_from": "pattern"}
    c_agree = scoring.compute_confidence(rag, rag, cb_agree, items, "alpha here", options)
    c_dis = scoring.compute_confidence(rag, rag, cb_disagree, items, "alpha here", options)
    assert c_agree["agreement"] == 1.0 and c_dis["agreement"] == 0.0
    assert c_agree["confidence"] > c_dis["confidence"]


def test_confidence_in_range():
    items = []
    options = {"A": "alpha"}
    primary = {"answer": "A", "parsed_from": "fallback"}
    c = scoring.compute_confidence(primary, primary, None, items, "", options)
    assert 0.0 <= c["confidence"] <= 1.0
    assert c["agreement"] is None
