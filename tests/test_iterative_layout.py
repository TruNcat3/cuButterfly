from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def layout_index(logical, stages, consumer_stage):
    low_mask = (1 << consumer_stage) - 1
    return ((logical & low_mask) |
            ((logical >> (consumer_stage + 1)) << consumer_stage) |
            (((logical >> consumer_stage) & 1) << (stages - 1)))


def test_writer_aligned_layout_is_a_permutation():
    for stages in range(2, 13):
        for consumer_stage in range(stages):
            mapped = [layout_index(i, stages, consumer_stage) for i in range(1 << stages)]
            assert sorted(mapped) == list(range(1 << stages))
