import numpy as np

def load_normalization_stats(path="models/normalization_stats.npz"):
    stats = np.load(path)
    return stats["mean"], stats["std"]
