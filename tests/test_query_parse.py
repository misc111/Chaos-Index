from src.query.parse import parse_question


def test_parse_question_team_variants_nhl_with_explicit_default():
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
        ("kings next game", "LAK"),
    ]

    for question, team in cases:
        intent = parse_question(question, default_league="NHL")
        assert intent.intent_type == "team_next_game"
        assert intent.team == team
        assert intent.league == "NHL"


def test_parse_question_team_variants_nba():
    cases = [
        ("raptors next game", "TOR"),
        ("what are the knicks odds tonight", "NYK"),
        ("thunder next game", "OKC"),
        ("boston celtics next game", "BOS"),
    ]

    for question, team in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_next_game"
        assert intent.team == team
        assert intent.league == "NBA"


def test_parse_question_default_league_override():
    nba_intent = parse_question("Toronto next game")
    nhl_intent = parse_question("Toronto next game", default_league="NHL")

    assert nba_intent.intent_type == "team_next_game"
    assert nba_intent.team == "TOR"
    assert nba_intent.league == "NBA"

    assert nhl_intent.intent_type == "team_next_game"
    assert nhl_intent.team == "TOR"
    assert nhl_intent.league == "NHL"


def test_parse_question_next_n_games():
    cases = [
        ("What are the Leafs odds in the next 3 games?", 3, "TOR", "NHL"),
        ("red wings next three games", 3, "DET", "NHL"),
        ("devils next couple", 2, "NJD", "NHL"),
        ("lightning next few", 3, "TBL", "NHL"),
        ("What are the Knicks odds in the next 3 games?", 3, "NYK", "NBA"),
        ("raptors next few", 3, "TOR", "NBA"),
    ]
    for question, n_games, team, league in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_next_n_games"
        assert intent.team == team
        assert intent.league == league
        assert intent.n_games == n_games


def test_parse_question_championships():
    cases = [
        ("what's the probability the kings win the stanley cup?", "LAK", "NHL", "Stanley Cup"),
        ("do the bolts win the cup?", "TBL", "NHL", "Stanley Cup"),
        ("new jersey to win it all", "NJD", "NHL", "Stanley Cup"),
        ("what are the odds the knicks win the nba finals?", "NYK", "NBA", "NBA Finals"),
        ("do the raptors win it all?", "TOR", "NBA", "NBA Finals"),
    ]
    for question, team, league, competition in cases:
        intent = parse_question(question)
        assert intent.intent_type == "team_championship"
        assert intent.team == team
        assert intent.league == league
        assert intent.competition == competition


def test_parse_question_best_model_still_supported():
    intent = parse_question("Which model has performed best the last 45 days?")
    assert intent.intent_type == "best_model"
    assert intent.window_days == 45


def test_parse_question_league_report():
    intent = parse_question("Give me the report of all teams in a table with next opponents")
    assert intent.intent_type == "league_report"
    assert intent.league == "NBA"

    nba_intent = parse_question("Give me the NBA team report table", default_league="NHL")
    assert nba_intent.intent_type == "league_report"
    assert nba_intent.league == "NBA"


def test_parse_question_bet_history_defaults_to_nba_and_tracks_game_breakdown():
    intent = parse_question("Tell me how much money I won or lost from last night's games. both in total and by games.")
    assert intent.intent_type == "bet_history_summary"
    assert intent.league == "NBA"
    assert intent.history_period == "yesterday"
    assert intent.include_games is True

    shorthand = parse_question("How much money did I win/lose last night?")
    assert shorthand.intent_type == "bet_history_summary"
    assert shorthand.league == "NBA"
    assert shorthand.history_period == "yesterday"
    assert shorthand.include_games is True

    recap = parse_question("How'd I do last night on my bets?")
    assert recap.intent_type == "bet_history_summary"
    assert recap.league == "NBA"
    assert recap.history_period == "yesterday"
    assert recap.include_games is True

    recap2 = parse_question("Recap my bets from yesterday")
    assert recap2.intent_type == "bet_history_summary"
    assert recap2.league == "NBA"
    assert recap2.history_period == "yesterday"
    assert recap2.include_games is True

    cumulative = parse_question("What are my cumulative net profits or losses and how much have I risked since the beginning of tracking?")
    assert cumulative.intent_type == "bet_history_summary"
    assert cumulative.league == "NBA"
    assert cumulative.history_period == "all_time"
    assert cumulative.include_games is False
