"""
LieGG-style generator extraction from a trained network.

Implements the method of Moskalev et al., "LieGG: Studying Learned Lie
Group Generators" (NeurIPS 2022): the infinitesimal-invariance condition

    d/deps F((I + eps*A) x) |_{eps=0}  =  grad F(x)^T A x  =  0

is linear in the entries of the generator ``A``, so stacking one row per
data sample yields the *network polarization matrix*

    E[i, j*n + k] = dF/dx_j(x_i) * x_i[k],        E in R^{N x n^2}.

The right singular vectors of ``E`` with near-zero singular values span
the Lie algebra of linear symmetries learned by the network.  Singular
value magnitudes quantify how invariant the network actually is
("symmetry variance"); the distance of an extracted generator to a known
true algebra is the "symmetry bias".

Two extensions beyond the strictly linear GL(n) action of the paper:

- ``action="translation"`` uses rows ``E[i, j] = dF/dx_j(x_i)`` whose
  null space contains the translation generators ``x -> x + eps*g``.
- ``action="affine"`` concatenates both blocks and extracts pairs
  ``(A, g)`` for the affine action ``x -> (I + eps*A) x + eps*g``.

All functions are torch-free at the interface (numpy in / numpy out)
except ``train_regressor`` / ``input_gradients``, which need torch.
"""

from typing import Callable, Optional, Sequence, Union

import numpy as np


# ------------------------------------------------------------------
# Network training and input gradients (torch)
# ------------------------------------------------------------------

def train_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dims: Sequence[int] = (64, 64),
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
):
    """Train a small Tanh MLP regressor F: R^n -> R for LieGG analysis.

    Returns the trained ``torch.nn.Module`` in eval mode.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    _device = torch.device(device)

    n_inputs = X.shape[1]
    layers, width_in = [], n_inputs
    for width in hidden_dims:
        layers += [nn.Linear(width_in, width), nn.Tanh()]
        width_in = width
    layers += [nn.Linear(width_in, 1)]
    model = nn.Sequential(*layers).to(_device)

    X_t = torch.tensor(X, dtype=torch.float32, device=_device)
    y_t = torch.tensor(y, dtype=torch.float32, device=_device).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  [train_regressor] epoch {epoch + 1}/{n_epochs}  mse={loss.item():.3e}")

    model.eval()
    return model


def input_gradients(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Gradients of the scalar network output w.r.t. each input, (N, n)."""
    import torch

    X_t = torch.tensor(X, dtype=torch.float32, device=torch.device(device))
    X_t.requires_grad_(True)
    out = model(X_t)
    if out.ndim > 1:
        out = out.squeeze(-1)
    grads = torch.autograd.grad(out.sum(), X_t)[0]
    return grads.detach().cpu().numpy().astype(float)


# ------------------------------------------------------------------
# Polarization matrix and generator extraction (numpy)
# ------------------------------------------------------------------

def polarization_matrix(
    grads: np.ndarray,
    X: np.ndarray,
    action: str = "linear",
) -> np.ndarray:
    """
    Build the network polarization matrix from per-sample input gradients.

    Parameters
    ----------
    grads : (N, n)
        Input gradients dF/dx evaluated at each sample.
    X : (N, n)
        The samples themselves.
    action : {"linear", "translation", "affine"}
        Group action whose infinitesimal-invariance rows are stacked:
        linear      -> E[i, j*n+k] = grads[i,j] * X[i,k]     (N, n^2)
        translation -> E[i, j]     = grads[i,j]              (N, n)
        affine      -> [linear block | translation block]    (N, n^2 + n)

    Returns
    -------
    E : np.ndarray
        Polarization matrix; its (approximate) null space is the learned
        Lie algebra for the chosen action.
    """
    grads = np.asarray(grads, dtype=float)
    X = np.asarray(X, dtype=float)
    n_samples, n_inputs = X.shape

    linear_block = np.einsum("ij,ik->ijk", grads, X).reshape(n_samples, n_inputs**2)

    if action == "linear":
        return linear_block
    elif action == "translation":
        return grads.copy()
    elif action == "affine":
        return np.hstack([linear_block, grads])
    else:
        raise ValueError(
            f"Unknown action '{action}'. Must be 'linear', 'translation', or 'affine'."
        )


def extract_liegg_generators(
    E: np.ndarray,
    n_inputs: int,
    action: str = "linear",
    n_generators: Optional[int] = None,
    rel_tol: float = 1e-2,
) -> dict:
    """
    SVD the polarization matrix and return the near-null-space generators.

    Parameters
    ----------
    E : polarization matrix from ``polarization_matrix``.
    n_inputs : int
        Input dimensionality n (needed to reshape vec(A) back to (n, n)).
    action : {"linear", "translation", "affine"}
        Must match the action used to build ``E``.
    n_generators : int, optional
        If given, take exactly this many smallest-singular-value vectors.
        If None, auto-detect: keep vectors with sigma < rel_tol * sigma_max.
    rel_tol : float
        Relative threshold for auto-detection.

    Returns
    -------
    dict with keys:
        spectrum       : (d,) all singular values, descending
        symmetry_variance : (k,) squared smallest singular values / N,
                            one per extracted generator (0 = exact symmetry)
        n_detected     : number of singular values under the threshold
        generators     : list of extracted generators —
                         linear      -> (n, n) matrices A
                         translation -> (n,)  vectors g
                         affine      -> (A, g) tuples
    """
    E = np.asarray(E, dtype=float)
    n_samples = E.shape[0]

    _, s, Vt = np.linalg.svd(E, full_matrices=True)
    d = Vt.shape[0]
    spectrum = np.zeros(d)
    spectrum[: len(s)] = s

    n_detected = int(np.sum(spectrum < rel_tol * spectrum[0]))
    k = n_generators if n_generators is not None else n_detected
    k = min(k, d)

    generators = []
    for i in range(d - k, d):
        v = Vt[i]
        if action == "linear":
            generators.append(v.reshape(n_inputs, n_inputs))
        elif action == "translation":
            generators.append(v.copy())
        else:  # affine
            A = v[: n_inputs**2].reshape(n_inputs, n_inputs)
            g = v[n_inputs**2 :]
            generators.append((A, g))
    generators = generators[::-1]  # smallest singular value first

    sym_variance = (spectrum[d - k : d][::-1] ** 2) / n_samples

    return {
        "spectrum": spectrum,
        "symmetry_variance": sym_variance,
        "n_detected": n_detected,
        "generators": generators,
    }


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def _as_flat_unit(gen: Union[np.ndarray, tuple]) -> np.ndarray:
    """Flatten a generator (matrix, vector, or affine pair) to a unit vector."""
    if isinstance(gen, tuple):
        flat = np.concatenate([np.ravel(part) for part in gen])
    else:
        flat = np.ravel(np.asarray(gen, dtype=float))
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def symmetry_bias(
    generators: Sequence[Union[np.ndarray, tuple]],
    true_generators: Sequence[Union[np.ndarray, tuple]],
) -> np.ndarray:
    """
    Residual of each extracted generator outside the true Lie algebra.

    Each generator is flattened and normalised; the true generators span a
    subspace onto which the extracted one is projected.  Returned per-
    generator bias is ``||g_hat - P_true(g_hat)||`` in [0, 1]:
    0 = generator lies exactly in the true algebra, 1 = orthogonal to it.
    """
    basis = np.stack([_as_flat_unit(t) for t in true_generators], axis=1)
    Q, _ = np.linalg.qr(basis)
    biases = []
    for gen in generators:
        v = _as_flat_unit(gen)
        residual = v - Q @ (Q.T @ v)
        biases.append(float(np.linalg.norm(residual)))
    return np.asarray(biases)


def orbit_invariance_error(
    func: Callable[[np.ndarray], np.ndarray],
    X0: np.ndarray,
    generator: Union[np.ndarray, tuple],
    action: str = "linear",
    tau_max: float = 0.3,
    n_steps: int = 20,
) -> float:
    """
    Relative deviation of ``func`` along the orbit of a generator.

    Propagates each row of ``X0`` with exp(tau*A) (linear), x + tau*g
    (translation), or exp(tau*A) x + tau*g (affine, first order) for
    tau in [-tau_max, tau_max] and returns

        mean_i  std_tau f(x_i(tau)) / (std_j f(x_j(0)) + 1e-12).

    A generator of a true symmetry of ``func`` gives a value near 0.
    """
    from scipy.linalg import expm

    X0 = np.atleast_2d(np.asarray(X0, dtype=float))
    taus = np.linspace(-tau_max, tau_max, n_steps + 1)

    values = np.empty((len(taus), X0.shape[0]))
    for t_idx, tau in enumerate(taus):
        if action == "linear":
            Xt = X0 @ expm(tau * np.asarray(generator)).T
        elif action == "translation":
            Xt = X0 + tau * np.asarray(generator)
        elif action == "affine":
            A, g = generator
            Xt = X0 @ expm(tau * np.asarray(A)).T + tau * np.asarray(g)
        else:
            raise ValueError(f"Unknown action '{action}'.")
        values[t_idx] = np.asarray(func(Xt), dtype=float)

    per_point_std = values.std(axis=0)
    baseline = float(values[len(taus) // 2].std()) + 1e-12
    return float(per_point_std.mean() / baseline)
