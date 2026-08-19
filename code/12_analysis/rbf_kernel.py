"""RBF kernel (Gaussian kernel) with median heuristic bandwidth selection.

Used by Exp 10 (cross-prompt representation alignment) for MMD computation.
Reference: Gretton et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
"""

import torch


def rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix K[i][j] = exp(-||x_i - x_j||^2 / (2 * sigma^2)).

    When sigma is None, uses the median heuristic: sigma^2 = median of all pairwise
    squared distances. The median is detached from the computation graph so the
    bandwidth itself receives no gradient.

    Args:
        x: (n, d) tensor of n samples in d dimensions.
        sigma: bandwidth. If None, auto-selected via median heuristic.

    Returns:
        (n, n) kernel matrix, values in (0, 1].
    """
    x2 = (x * x).sum(1, keepdim=True)                # (n, 1)  squared norms
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())          # (n, n)  ||x_i - x_j||^2
    dist2 = dist2.clamp(min=0.0)

    if sigma is None:
        with torch.no_grad():
            nonzero = dist2[dist2 > 0]
            med = torch.median(nonzero) if nonzero.numel() > 0 else torch.tensor(1.0, device=x.device)
            sigma2 = med.clamp(min=1e-6)
    else:
        sigma2 = torch.tensor(sigma ** 2, device=x.device)

    return torch.exp(-dist2 / (2.0 * sigma2))


def _median_bandwidth2(z: torch.Tensor) -> torch.Tensor:
    """Return the median pairwise squared distance of z (scalar tensor)."""
    z2 = (z * z).sum(1, keepdim=True)
    dist2 = z2 + z2.t() - 2.0 * (z @ z.t())
    dist2 = dist2.clamp(min=0.0)
    with torch.no_grad():
        nonzero = dist2[dist2 > 0]
        med = torch.median(nonzero) if nonzero.numel() > 0 else torch.tensor(1.0, device=z.device)
        return med.clamp(min=1e-6)


def rbf_cross(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Kernel matrix between x and y: K[i,j] = exp(-||x_i - y_j||^2 / (2*sigma^2))."""
    x2 = (x * x).sum(1, keepdim=True)
    y2 = (y * y).sum(1, keepdim=True)
    dist2 = x2 + y2.t() - 2.0 * (x @ y.t())
    dist2 = dist2.clamp(min=0.0)
    sigma2 = torch.tensor(sigma ** 2, device=x.device)
    return torch.exp(-dist2 / (2.0 * sigma2))


def mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Unbiased estimate of squared MMD (U-statistic) with RBF kernel.

    MMD^2 = 1/(n(n-1)) * sum_{i!=j} k(x_i, x_j)
          + 1/(m(m-1)) * sum_{i!=j} k(y_i, y_j)
          - 2/(n*m) * sum_ij k(x_i, y_j)

    When sigma is None, a single median-heuristic bandwidth is estimated
    from the pooled sample [x; y] and used for all three kernel matrices.

    Args:
        x: (n, d) samples from distribution P.
        y: (m, d) samples from distribution Q.
        sigma: RBF bandwidth (None = median heuristic).

    Returns:
        Scalar MMD^2 value.
    """
    n, m = x.shape[0], y.shape[0]
    if sigma is None:
        sigma2 = _median_bandwidth2(torch.cat([x, y], dim=0))
        sigma = float(torch.sqrt(sigma2).item())
    K_xx = rbf_kernel(x, sigma)
    K_yy = rbf_kernel(y, sigma)
    K_xy = rbf_cross(x, y, sigma)

    term_xx = (K_xx.sum() - torch.diagonal(K_xx).sum()) / (n * (n - 1))
    term_yy = (K_yy.sum() - torch.diagonal(K_yy).sum()) / (m * (m - 1))
    term_xy = K_xy.sum() / (n * m)

    return term_xx + term_yy - 2.0 * term_xy
