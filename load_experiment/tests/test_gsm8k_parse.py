from load_experiment.gsm8k_parse import gsm8k_rows, numbers_in, parse_cot
from load_experiment.load import min_load

DUCKS = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
         "muffins for her friends every day with four. She sells the remainder at the farmers' market "
         "daily for $2 per fresh duck egg. How much in dollars does she make every day?")


def test_chain_from_the_dataset():
    p = parse_cot("<<16-3-4=9>> <<9*2=18>>", DUCKS)
    assert p["uses"] == [set(), {0}]
    assert p["results"] == [9.0, 18.0]
    assert not p["ambiguous"]
    assert min_load(p["uses"]) == 1


def test_wide():
    p = parse_cot("<<4*6=24>> <<5*2=10>> <<24-10=14>>", "4 boxes of 6 pens and 5 boxes of 2 pencils")
    assert p["uses"] == [set(), set(), {0, 1}]
    assert min_load(p["uses"]) == 2


def test_ambiguous_when_a_result_also_appears_in_the_question():
    p = parse_cot("<<4*6=24>> <<24+24=48>>", "He has 24 apples and 4 bags of 6 apples")
    assert p["ambiguous"]


def test_numbers_and_number_words():
    assert numbers_in("$1,200 and 3.5 kg.") == [1200.0, 3.5]
    assert 3.0 in numbers_in("She eats three", words=True)


def test_unreadable_equation():
    assert parse_cot("<<2+2=x>>") is None


def test_rows():
    data = [dict(question=DUCKS, cot="<<16-3-4=9>> <<9*2=18>>", answer="18")]
    (row,) = gsm8k_rows(data)
    assert row["steps"] == 2 and row["load"] == 1 and row["gold"] == 18.0
