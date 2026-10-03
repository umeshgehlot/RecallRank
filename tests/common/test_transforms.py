import math

from recsys_common.transforms import (
    log_compress,
    pad_and_mask,
    quantile_bucket,
    vocab_lookup,
    z_score,
)


class TestLogCompress:
    def test_positive_value(self):
        assert log_compress(99.0) == math.log(100.0)

    def test_zero(self):
        assert log_compress(0.0) == 0.0

    def test_negative_clamps_to_zero(self):
        assert log_compress(-5.0) == 0.0

    def test_small_positive(self):
        assert log_compress(0.001) == math.log(1.001)


class TestZScore:
    def test_standard_case(self):
        assert z_score(10.0, mu=5.0, sigma=2.0) == 2.5

    def test_zero_mean(self):
        assert z_score(3.0, mu=0.0, sigma=1.0) == 3.0

    def test_zero_sigma_returns_zero(self):
        assert z_score(10.0, mu=5.0, sigma=0.0) == 0.0


class TestQuantileBucket:
    def test_middle_bucket(self):
        boundaries = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        assert quantile_bucket(25.0, boundaries) == 2

    def test_below_all_boundaries(self):
        boundaries = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        assert quantile_bucket(5.0, boundaries) == 0

    def test_above_all_boundaries(self):
        boundaries = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        assert quantile_bucket(95.0, boundaries) == 9

    def test_exact_boundary(self):
        boundaries = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        assert quantile_bucket(30.0, boundaries) == 3


class TestVocabLookup:
    def test_known_token(self):
        vocab = {"apple": 1, "banana": 2}
        assert vocab_lookup("apple", vocab) == 1

    def test_unknown_token_returns_zero(self):
        vocab = {"apple": 1, "banana": 2}
        assert vocab_lookup("cherry", vocab) == 0

    def test_empty_string_returns_zero(self):
        vocab = {"apple": 1}
        assert vocab_lookup("", vocab) == 0


class TestPadAndMask:
    def test_shorter_than_max(self):
        padded, mask = pad_and_mask([1, 2], 5)
        assert padded == [1, 2, 0, 0, 0]
        assert mask == [1, 1, 0, 0, 0]

    def test_exact_length(self):
        padded, mask = pad_and_mask([1, 2, 3], 3)
        assert padded == [1, 2, 3]
        assert mask == [1, 1, 1]

    def test_longer_than_max_truncates(self):
        padded, mask = pad_and_mask([1, 2, 3, 4, 5], 3)
        assert padded == [1, 2, 3]
        assert mask == [1, 1, 1]

    def test_empty_list(self):
        padded, mask = pad_and_mask([], 3)
        assert padded == [0, 0, 0]
        assert mask == [0, 0, 0]
