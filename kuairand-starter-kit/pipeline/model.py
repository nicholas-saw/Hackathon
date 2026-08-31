"""Model architecture -- the only place model/loss changes are allowed. See AGENT_RULES.md.

Public contract (train.py depends on these):
    FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)
        .step(X, y) -> loss:float      # one mini-batch gradient update (pointwise BCE)
        .step_list(X, y, group_ends, lw_alpha=0.3) -> loss:float
            # X, y must be pre-sorted so that each user's rows are contiguous.
            # group_ends: 1D int array of exclusive end offsets into X/y marking each
            # user group (e.g. [3, 3, 5] means groups [0:3], [3:3] (empty, skipped),
            # [3:5]). Within-group softmax (ListNet top-1) mixed with lw_alpha * BCE.
        .predict(X, bs=200_000) -> np.ndarray[float]   # scores; higher = more relevant
        .V, .W, .b                     # used by train.py to save/restore the best checkpoint
"""
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _apply_grad(self, X, g, S, E):
        """Shared Adam update given per-row dLoss/dlogit values g."""
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        self._apply_grad(X, g, S, E)
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_list(self, X, y, group_ends, lw_alpha=0.3):
        """Within-user listwise softmax (ListNet top-1), mixed with pointwise BCE.

        X, y must already be sorted so each user's rows are contiguous; group_ends
        gives the exclusive end index of each contiguous group. Groups with zero
        positives contribute no listwise gradient (target distribution p_i = y_i /
        sum(y) is undefined) but still contribute their BCE term, so all-negative
        users still push item-side signal.

        Loss = mean_over_active_groups(-sum_i p_i log q_i) + lw_alpha * mean_BCE_all_rows
        """
        B = len(y)
        z, E, S = self.logits(X)
        p_sig = sigmoid(z)
        g_bce = (lw_alpha * (p_sig - y) / B).astype(np.float32)

        g_list = np.zeros(B, dtype=np.float32)
        start = 0
        n_active = 0
        total_listwise_loss = 0.0
        for end in group_ends:
            if end > start:
                y_g = y[start:end]
                s_g = y_g.sum()
                if s_g > 0:
                    z_g = z[start:end]
                    zg = z_g - z_g.max()
                    ez = np.exp(zg)
                    q = ez / ez.sum()
                    p_g = y_g / s_g
                    g_list[start:end] = (q - p_g)
                    n_active += 1
                    total_listwise_loss += float(-np.sum(p_g * np.log(q + 1e-9)))
            start = end
        if n_active > 0:
            g_list /= n_active

        g = (g_list + g_bce).astype(np.float32)
        self._apply_grad(X, g, S, E)

        bce_loss = float(-np.mean(y * np.log(p_sig + 1e-9) + (1 - y) * np.log(1 - p_sig + 1e-9)))
        avg_list_loss = total_listwise_loss / max(n_active, 1)
        return avg_list_loss + lw_alpha * bce_loss

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])
