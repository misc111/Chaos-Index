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


def test_parse_question_next_n_games():
    cases = [
        ("What are Toronto's odds in the next 3 games?", 3, "TOR"),
        ("red wings next three games", 3, "DET"),
        ("devils next couple", 2, "NJD"),
        ("lightning next few", 3, "TBL"),
    ]
    for question, n_games, team in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_next_n_games"
        assert intent.team == team
        assert intent.n_games == n_games


def test_parse_question_stanley_cup():
    cases = [
        ("what's the probability the kings win the stanley cup?", "LAK"),
        ("do the bolts win the cup?", "TBL"),
        ("new jersey to win it all", "NJD"),
    ]
    for question, team in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_stanley_cup"
        assert intent.team == team


def test_parse_question_best_model_still_supported():
    intent = parse_question("Which model has performed best the last 45 days?")
    assert intent.intent_type == "best_model"
    assert intent.window_days == 45
