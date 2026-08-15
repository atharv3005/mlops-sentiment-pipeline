from utils import clean_text


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("This Movie WAS Great!!!") == "this movie was great"


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many    spaces") == "too many spaces"


def test_clean_text_handles_numbers():
    assert clean_text("Rated 10/10 would watch again") == "rated 10 10 would watch again"
