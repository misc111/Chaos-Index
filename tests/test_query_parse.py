from src.query.parse import parse_question


def test_parse_question_team_variants():
    cases = [
        ("new jersey", "NJD"),
        ("What are New Jersey's chances?", "NJD"),
        ("devils next game", "NJD"),
        ("lightning tonight", "TBL"),
        ("how are the bolts looking", "TBL"),
        ("tampa bay", "TBL"),
        ("toronto", "TOR"),
        ("leafs next one", "TOR"),
        ("habs", "MTL"),
        ("ARI next game", "UTA"),
    ]

    for question, team in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_next_game"
        assert intent.team == team


def test_parse_question_best_model_still_supported():
    intent = parse_question("Which model has performed best the last 45 days?")
    assert intent.intent_type == "best_model"
    assert intent.window_days == 45
