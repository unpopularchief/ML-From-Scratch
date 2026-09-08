r"""Gaussian naive Bayes — a generative classifier fitted by closed-form MLE.

Models each class's features as independent Gaussians,
:math:`P(x \mid c) = \prod_j \mathcal{N}(x_j; \mu_{cj}, \sigma^2_{cj})`, and
classifies by the posterior via Bayes' rule. Fitting is a single pass: the
maximum-likelihood mean and (biased) variance of every feature within every
class, plus the class priors.

.. math::
    \mathrm{jll}(x, c) = \log \pi_c
      - \frac12 \sum_{j=1}^{d}\left[\log(2\pi\sigma^2_{cj})
        + \frac{(x_j - \mu_{cj})^2}{\sigma^2_{cj}}\right]

:meth:`predict` returns :math:`\arg\max_c \mathrm{jll}(x, c)`;
:meth:`predict_proba` normalises with ``logsumexp`` over the classes.

A variance floor ``epsilon_ = var_smoothing * max_j Var[X[:, j]]`` is added to
every variance so a feature that is constant within a class does not divide by
zero. This matches ``sklearn.naive_bayes.GaussianNB``.

Full derivation: ``docs/derivations/gaussian_nb.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.math import logsumexp
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y


def _gaussian_log_density(
    X: FloatArray, mean: FloatArray, var: FloatArray
) -> FloatArray:
    r"""Per-feature Gaussian log-density :math:`\log \mathcal{N}(x_j; \mu, \sigma^2)`.

    ``X`` is ``(n, d)``; ``mean`` and ``var`` are ``(K, d)``. Returns the
    ``(n, K, d)`` array whose ``[i, c, j]`` entry is
    :math:`-\tfrac12[\log(2\pi\sigma^2_{cj}) + (x_{ij} - \mu_{cj})^2 /
    \sigma^2_{cj}]`.
    """
    # broadcast X to (n, 1, d) against (K, d) -> (n, K, d)
    residual_sq = (X[:, None, :] - mean) ** 2  # (x_ij − μ_cj)²
    return -0.5 * (np.log(2.0 * np.pi * var) + residual_sq / var)


def _joint_log_likelihood(
    X: FloatArray, mean: FloatArray, var: FloatArray, log_prior: FloatArray
) -> FloatArray:
    r"""Unnormalised log-posterior :math:`\mathrm{jll}(x, c)`, shape ``(n, K)``.

    :math:`\mathrm{jll}(x, c) = \log \pi_c
    + \sum_j \log \mathcal{N}(x_j; \mu_{cj}, \sigma^2_{cj})` — the log of the
    numerator of Bayes' rule. ``log_prior`` is :math:`\log \pi_c`, shape
    ``(K,)``.
    """
    # Σ_j log N(x_j; μ_cj, σ²_cj)  over the d features
    log_likelihood = _gaussian_log_density(X, mean, var).sum(axis=2)  # (n, K)
    return log_prior + log_likelihood  # + log π_c


class GaussianNB(Estimator):
    r"""Gaussian naive Bayes classifier.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), optional
        Prior probability of each class, ordered as ``classes_`` (the sorted
        unique labels). Must be non-negative and sum to 1. If ``None`` (the
        default), the priors are the maximum-likelihood class frequencies
        ``class_count_ / n_samples``.
    var_smoothing : float, default=1e-9
        Fraction of the largest feature variance (across the whole training
        set) added to every class-conditional variance for numerical
        stability. ``epsilon_`` stores the resulting absolute value. Matches
        ``sklearn.naive_bayes.GaussianNB``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels seen in ``y``. ``predict`` returns values from
        here; ``predict_proba`` columns are in this order.
    class_count_ : ndarray of shape (n_classes,)
        Number of training samples in each class.
    class_prior_ : ndarray of shape (n_classes,)
        Prior probability of each class (supplied ``priors`` or the MLE
        frequencies).
    mean_ : ndarray of shape (n_classes, n_features)
        Per-class, per-feature mean :math:`\mu_{cj}` (sklearn calls this
        ``theta_``).
    var_ : ndarray of shape (n_classes, n_features)
        Per-class, per-feature variance :math:`\sigma^2_{cj}` — the biased
        MLE (divided by the class count), plus ``epsilon_``.
    epsilon_ : float
        Absolute additive value ``var_smoothing * max_j Var[X[:, j]]``.

    Notes
    -----
    ``fit`` is one closed-form pass: for each class ``c`` and feature ``j``,
    :math:`\hat\mu_{cj}` and :math:`\hat\sigma^2_{cj}` are the sample mean and
    (biased) sample variance of that feature over the in-class rows. There is
    no objective to minimise. Prediction works in log-space:
    ``predict`` takes the argmax of the joint log-likelihood
    :math:`\mathrm{jll}(x, c) = \log \pi_c + \sum_j \log
    \mathcal{N}(x_j; \mu_{cj}, \sigma^2_{cj})` and ``predict_proba``
    normalises it with ``logsumexp`` over the classes. See
    ``docs/derivations/gaussian_nb.md``.

    The "naive" conditional-independence assumption makes ``predict_proba``
    overconfident when features are correlated, though the ``predict`` argmax
    is often still right. GaussianNB is equivariant to a per-feature affine
    rescaling of ``X``, so — unlike KNN — it does not need standardised
    inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.naive_bayes import GaussianNB
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = GaussianNB().fit(X, y)
    >>> model.predict(np.array([[-1.5], [1.5]])).tolist()
    [0.0, 1.0]

    """

    def __init__(
        self,
        *,
        priors: object = None,
        var_smoothing: float = 1e-9,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X: FeatureMatrix, y: TargetVector) -> GaussianNB:
        """Fit the per-class Gaussians and priors to ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``var_smoothing`` is negative, or ``priors`` is given and is
            not a non-negative length-``n_classes`` vector summing to 1.

        """
        if self.var_smoothing < 0:
            raise ValueError(f"var_smoothing must be >= 0, got {self.var_smoothing}.")

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        supplied_prior = self._validated_priors(n_classes)

        # Variance floor: a fraction of the largest feature variance over the
        # whole training set, so it scales with the data (sklearn's rule).
        self.epsilon_ = self.var_smoothing * float(X.var(axis=0).max())

        self.class_count_ = np.zeros(n_classes)
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        for k, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[k] = X_c.shape[0]
            self.mean_[k] = X_c.mean(axis=0)  # μ_cj = (1/N_c) Σ x_ij
            self.var_[k] = X_c.var(axis=0)  # σ²_cj = (1/N_c) Σ (x_ij − μ_cj)²
        self.var_ += self.epsilon_  # variance floor

        if supplied_prior is not None:
            self.class_prior_ = supplied_prior
        else:
            self.class_prior_ = self.class_count_ / n_samples  # π_c = N_c / n
        return self

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        Returns ``classes_[argmax_c jll(x, c)]``; the evidence ``log P(x)``
        is a per-row constant and is not formed. Ties resolve to the lowest
        class label (:func:`numpy.argmax` returns the first maximal entry and
        ``classes_`` is sorted).

        Returns
        -------
        ndarray of shape (n_samples,)
            Each entry is one of the values in ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        # into a local first — it runs the fitted / feature-count checks, so
        # ``self.classes_`` below can't raise a bare AttributeError pre-fit.
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return log class probabilities, shape ``(n_samples, n_classes)``.

        ``jll(x, c) - logsumexp_c' jll(x, c')`` — the joint log-likelihood
        normalised over the classes, where the subtracted term is
        ``log P(x)``. Columns are ordered as ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        jll = self._joint_log_likelihood(X)  # (n, K)
        # log P(x) = logsumexp_c jll(x, c), broadcast back over the columns
        return jll - logsumexp(jll, axis=1)[:, None]

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class probabilities, shape ``(n_samples, n_classes)``.

        ``exp(predict_log_proba(X))``. Rows sum to 1; columns are ordered as
        ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        return np.exp(self.predict_log_proba(X))

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.accuracy_score`: the fraction of
        samples whose predicted label matches the true label.
        """
        return accuracy_score(y, self.predict(X))

    def _joint_log_likelihood(self, X: FeatureMatrix) -> FloatArray:
        """Validate ``X`` against the fitted model, then return ``jll(x, c)``.

        Runs the fitted check and the feature-count check first, so every
        public predictor inherits them.
        """
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.mean_.shape[1]}."
            )
        return _joint_log_likelihood(
            X, self.mean_, self.var_, np.log(self.class_prior_)
        )

    def _validated_priors(self, n_classes: int) -> FloatArray | None:
        """Coerce and check the ``priors`` hyperparameter, or return ``None``.

        ``None`` means "use the MLE class frequencies". Otherwise the vector
        must be non-negative, length ``n_classes``, and sum to 1.
        """
        if self.priors is None:
            return None
        priors = np.asarray(self.priors, dtype=np.float64)
        if priors.shape != (n_classes,):
            raise ValueError(
                f"priors must have shape ({n_classes},) to match the number of "
                f"classes, got {priors.shape}."
            )
        if np.any(priors < 0.0):
            raise ValueError("priors must be non-negative.")
        if not np.isclose(priors.sum(), 1.0):
            raise ValueError(f"priors must sum to 1, got {float(priors.sum())}.")
        return priors
