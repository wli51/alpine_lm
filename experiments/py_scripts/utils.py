import random

def deterministic_seeds(master_seed, n):
    rng = random.Random(master_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(n)]
