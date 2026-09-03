"""Synthetic dataset generators for tests and examples.

Every generator takes an explicit ``random_state`` and returns plain
NumPy arrays — no on-disk caching, no real data. Real datasets (MNIST,
tiny-shakespeare) are added as loaders in this same package starting at
M3, downloading into a gitignored cache directory on first use rather
than being committed to the repository.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FeatureMatrix, IntArray, TargetVector
from scratchgrad.utils.validation import check_random_state


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FeatureMatrix, TargetVector]:
    r"""Generate a linear regression problem, :math:`y = Xw + b + \varepsilon`.

    ``X`` is drawn from a standard normal distribution. The true weights
    ``w`` and bias ``b`` are drawn once from a standard normal too, then
    fixed for every sample; :math:`\varepsilon \sim \mathcal{N}(0,
    \mathrm{noise}^2)` is added independently per sample.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=1
        Number of features (dimensionality of ``X``).
    noise : float, default=0.0
        Standard deviation of the Gaussian noise added to ``y``. 0 gives
        an exactly linear relationship — useful for testing a solver
        against the closed-form solution with no estimation error.
    random_state : int, numpy.random.Generator, or None
        Seed controlling every random draw below. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    """
    rng = check_random_state(random_state)
    X = rng.standard_normal((n_samples, n_features))
    true_w = rng.standard_normal(n_features)
    true_b = rng.standard_normal()
    y = X @ true_w + true_b + rng.normal(0.0, noise, size=n_samples)
    return X, y


def make_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FeatureMatrix, IntArray]:
    r"""Generate isotropic Gaussian clusters for classification/clustering.

    ``centers`` cluster centers are drawn uniformly from
    :math:`[-10, 10]^{\text{n\_features}}`; samples are drawn around the
    center assigned to them as :math:`x \sim \mathcal{N}(\text{center},
    \text{cluster\_std}^2 I)`. Samples are split as evenly as possible
    across centers.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate, split across ``centers`` clusters.
    n_features : int, default=2
        Number of features (dimensionality of ``X``).
    centers : int, default=3
        Number of cluster centers (and so the number of classes in ``y``).
    cluster_std : float, default=1.0
        Standard deviation of each cluster.
    random_state : int, numpy.random.Generator, or None
        Seed controlling every random draw below. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,), dtype int64
        Cluster/class index in ``[0, centers)`` for each sample.

    """
    rng = check_random_state(random_state)
    center_points = rng.uniform(-10.0, 10.0, size=(centers, n_features))

    # Split n_samples as evenly as possible across `centers` clusters, e.g.
    # n_samples=10, centers=3 -> cluster sizes [4, 3, 3].
    base_size, remainder = divmod(n_samples, centers)
    cluster_sizes = [
        base_size + 1 if i < remainder else base_size for i in range(centers)
    ]

    X_parts, y_parts = [], []
    for cluster_index, size in enumerate(cluster_sizes):
        X_parts.append(
            rng.normal(
                loc=center_points[cluster_index],
                scale=cluster_std,
                size=(size, n_features),
            )
        )
        y_parts.append(np.full(size, cluster_index, dtype=np.int64))

    return np.concatenate(X_parts), np.concatenate(y_parts)


def make_moons(
    n_samples: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FeatureMatrix, IntArray]:
    r"""Generate two interleaving half-circles ("moons").

    A classic non-linearly-separable binary classification toy set: class
    0 traces the upper half of a unit circle, :math:`(\cos\theta,
    \sin\theta)` for :math:`\theta \in [0, \pi]`; class 1 traces the lower
    half of a circle offset by ``(1, -0.5)`` and flipped vertically,
    :math:`(1 - \cos\theta, 1 - \sin\theta - 0.5)`, so the two arcs
    interleave rather than sitting concentrically.

    Parameters
    ----------
    n_samples : int, default=100
        Split as evenly as possible between the two moons.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to each point.
    random_state : int, numpy.random.Generator, or None
        Seed controlling every random draw below. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,), dtype int64
        0 for the upper moon, 1 for the lower moon.

    """
    rng = check_random_state(random_state)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    theta_upper = rng.uniform(0.0, np.pi, size=n_upper)
    upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])

    theta_lower = rng.uniform(0.0, np.pi, size=n_lower)
    lower = np.column_stack(
        [1.0 - np.cos(theta_lower), 1.0 - np.sin(theta_lower) - 0.5]
    )

    X = np.concatenate([upper, lower])
    X += rng.normal(0.0, noise, size=X.shape)
    y = np.concatenate(
        [np.zeros(n_upper, dtype=np.int64), np.ones(n_lower, dtype=np.int64)]
    )
    return X, y
