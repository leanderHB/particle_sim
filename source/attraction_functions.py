def F_LJ(r, r0, epsilon=1):  # Lennard Jones potential
    sigma = 0.5 ** (1 / 6) * r0
    s_over_r = sigma / r
    return -4 * epsilon * (7 * s_over_r**7 - 13 * s_over_r**13) / sigma
