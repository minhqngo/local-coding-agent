def safe_divide(a, b):
    # BUG: raises ZeroDivisionError instead of returning 0.0 for a zero divisor.
    return a / b
