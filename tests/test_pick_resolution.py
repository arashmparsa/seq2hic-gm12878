from seq2hic.targets import pick_resolution


def test_picks_coarsest_at_or_below_preferred():
    assert pick_resolution([1, 1000, 5000, 10000, 2500000], 5000) == 5000
    assert pick_resolution([1, 1000, 10000], 5000) == 1000


def test_falls_back_to_finest_when_all_coarser():
    assert pick_resolution([10000, 25000], 5000) == 10000
