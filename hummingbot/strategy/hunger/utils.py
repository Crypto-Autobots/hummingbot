from math import log10


def round_non_zero(num, digits=2):
    if num == 0:
        return 0
    if -1 < num < 1:
        dist = round(log10(abs(num))) - 1
        return round(num, max(digits, abs(dist)))
    return round(num, digits)
