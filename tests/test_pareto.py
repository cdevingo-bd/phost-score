import math

import pytest

from phost_score import Objective, dominates, ranks

SPEED, ERROR = Objective("speedup", "max"), Objective("err", "min")
BOTH = (SPEED, ERROR)


def p(speedup, err):
    return dict(speedup=speedup, err=err)


def test_better_on_one_and_equal_on_the_other_dominates():
    assert dominates(p(3, 1), p(2, 1), BOTH) and not dominates(p(2, 1), p(3, 1), BOTH)


def test_a_trade_off_dominates_neither_way():
    assert not dominates(p(3, 2), p(2, 1), BOTH) and not dominates(p(2, 1), p(3, 2), BOTH)


def test_equal_points_do_not_dominate_each_other():
    assert not dominates(p(2, 1), p(2, 1), BOTH)


def test_sense_decides_which_direction_is_better():
    assert dominates(p(1, 0.1), p(1, 0.2), BOTH)           # smaller error wins
    assert dominates(p(2, 1), p(1, 1), BOTH)               # larger speedup wins


def test_ranks_layer_the_fronts():
    points = [p(3, 3), p(2, 2), p(1, 1), p(2, 3), p(1, 3)]
    # front: (3,3) fast, (1,1) accurate, (2,2) in between; (2,3) is beaten by (3,3) and (2,2); (1,3) by all of them
    assert ranks(points, BOTH) == [0, 0, 0, 1, 2]


def test_equal_points_share_a_rank():
    assert ranks([p(2, 1), p(2, 1), p(1, 2)], BOTH) == [0, 0, 1]


def test_one_objective_ranks_by_order():
    assert ranks([dict(speedup=1), dict(speedup=3), dict(speedup=2)], (SPEED,)) == [2, 0, 1]


def test_no_points_and_no_objectives():
    assert ranks([], BOTH) == []
    with pytest.raises(ValueError, match="objective"):
        ranks([p(1, 1)], ())


@pytest.mark.parametrize("bad", [math.nan, math.inf, None, "fast"])
def test_a_bad_value_is_an_error_not_a_silent_loss(bad):
    with pytest.raises(ValueError, match="finite"):
        ranks([p(1, 1), dict(speedup=bad, err=1)], BOTH)
    with pytest.raises(ValueError, match="finite"):
        dominates(p(1, 1), dict(speedup=bad, err=1), BOTH)


def test_a_missing_objective_is_an_error():
    with pytest.raises(ValueError, match="missing"):
        ranks([dict(speedup=1)], BOTH)


def test_an_objective_needs_a_known_sense():
    with pytest.raises(ValueError, match="sense"):
        Objective("x", "bigger")
