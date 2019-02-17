import random
import numpy as np

def eval_map(y_true, y_pred, rel_threshold=0):
    """
        Code to calculate MAP (Mean Average Precision)
    """
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
            break
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s
