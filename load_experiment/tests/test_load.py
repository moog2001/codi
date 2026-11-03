import pytest

from load_experiment.load import min_load, order_load


def test_examples_from_the_plan():
    assert min_load([set(), {0}, {1}]) == 1                                  # chain
    assert min_load([set(), set(), {0, 1}]) == 2                             # wide
    assert min_load([set(), {0}]) == 1                                       # wide + hint
    assert min_load([set(), {0}, {1}, {2}, {3}]) == 1                        # 5-step chain
    assert min_load([set(), {0}, set(), {2}, {1, 3}]) == 2                   # 5-step wide
    assert min_load([set(), set(), {0, 1}, set(), set(), {3, 4}, {2, 5}]) == 3  # balanced tree


def test_order_matters_but_min_load_takes_the_best_order():
    # ((a*b)+(c*d)) + (e*f), written as: ab, cd, ef, then two additions
    uses = [set(), set(), set(), {0, 1}, {3, 2}]
    assert order_load([0, 1, 2, 3, 4], uses) == 3     # all three products first
    assert order_load([0, 1, 3, 2, 4], uses) == 2     # add as soon as possible
    assert min_load(uses) == 2


def test_no_steps_and_independent_steps():
    assert min_load([]) == 0
    assert min_load([set()]) == 0                    # one step, nothing held


def test_invalid_inputs():
    with pytest.raises(ValueError):
        min_load([{1}, {0}])                         # cycle
    with pytest.raises(ValueError):
        min_load([{5}])                              # reads a step that doesn't exist
    with pytest.raises(ValueError):
        order_load([1, 0], [set(), {0}])             # runs a step before its input
