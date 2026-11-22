

def test_boundary_f1_is_scale_invariant():
    """A prediction that is the truth shrunk toward its mean (what MSE training
    produces) must still recover the truth's boundaries."""
    import numpy as np
    from seq2hic import metrics as M

    n = 128
    idx = np.arange(n)
    dom = idx // 32
    t = np.where(dom[:, None] == dom[None, :], 1.0, -1.0)
    t = t - 0.01 * np.abs(idx[:, None] - idx[None, :]) + 0.003 * idx[:, None]
    m = np.ones_like(t, dtype=bool)
    shrunk = 0.3 * t
    assert M.boundary_f1(t, t, m, prominence=0.3)["boundary_f1"] == 1.0
    assert M.boundary_f1(shrunk, t, m, prominence=0.3)["boundary_f1"] == 1.0
