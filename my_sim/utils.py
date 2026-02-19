import numpy as np
import random

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
