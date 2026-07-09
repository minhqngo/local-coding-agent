from calc import safe_divide


def test_normal():
    assert safe_divide(10, 2) == 5


def test_zero_divisor():
    assert safe_divide(1, 0) == 0.0
