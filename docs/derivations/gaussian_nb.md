# Gaussian naive Bayes

The finalized derivation walkthrough for `scratchgrad.naive_bayes.GaussianNB`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and mirrors the `linear/` and
[`knn.md`](knn.md) walkthroughs.

Naive Bayes is a **generative** classifier: instead of modelling the decision
boundary directly (logistic regression) or memorising neighbours (KNN), it
models how each class *generates* its features, then inverts that with Bayes'
rule. The fit is a single closed-form pass — per-class means and variances —
so there is no objective to minimise and no gradient tier; the correctness
check is the closed form on hand data plus an independent recomputation of the
log-posterior.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | training design matrix |
| $y$ | $(n,)$ | training labels — any number of classes |
| $\mathcal{C}$ | $(K,)$ | `classes_` $= \mathrm{np.unique}(y)$, sorted |
| $N_c$ | scalar | number of training rows with $y_i = c$ (`class_count_`) |
| $\mu_{cj}$ | $(K, d)$ | per-class, per-feature mean (`mean_`) |
| $\sigma^2_{cj}$ | $(K, d)$ | per-class, per-feature variance (`var_`) |
| $\pi_c$ | $(K,)$ | class prior $P(y = c)$ (`class_prior_`) |
| $\varepsilon$ | scalar | additive variance floor (`epsilon_`) |

`classes_` is `np.unique(y)` (sorted); predictions are drawn from it.
GaussianNB is naturally multiclass — there is no binary restriction (unlike
`LogisticRegression`).

## 2. The model — Bayes' rule plus two assumptions

A generative classifier models the joint $P(x, c) = P(c)\,P(x \mid c)$ and
classifies by the posterior:

$$P(c \mid x) = \frac{P(c)\,P(x \mid c)}{\sum_{c'} P(c')\,P(x \mid c')}$$

Two modelling choices turn this into Gaussian naive Bayes.

**(a) The "naive" conditional-independence assumption.** Given the class, the
features are assumed independent:

$$P(x \mid c) = \prod_{j=1}^{d} P(x_j \mid c)$$

This is almost never literally true, but it replaces one $d$-dimensional
density estimate with $d$ one-dimensional ones — few parameters, no curse of
dimensionality, and the *argmax* is often right even when the probabilities
themselves come out overconfident.

**(b) Gaussian class-conditionals.** Each 1-D factor is normal:

$$P(x_j \mid c) = \mathcal{N}(x_j;\ \mu_{cj}, \sigma^2_{cj})
  = \frac{1}{\sqrt{2\pi\sigma^2_{cj}}}
    \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}\right)$$

So the whole model is $2Kd$ numbers ($\mu_{cj}$, $\sigma^2_{cj}$) plus $K$
priors.

## 3. Fitting — maximum likelihood, closed form

The parameters decouple per class and per feature. For a fixed $c, j$, the
log-likelihood of the $N_c$ in-class values is

$$\ell(\mu_{cj}, \sigma^2_{cj}) = \sum_{i:\,y_i = c}
  \left[ -\tfrac12\log(2\pi\sigma^2_{cj})
         - \frac{(x_{ij} - \mu_{cj})^2}{2\sigma^2_{cj}} \right]$$

Setting the partials to zero:

$$\frac{\partial \ell}{\partial \mu_{cj}}
  = \sum_{i:\,y_i=c} \frac{x_{ij} - \mu_{cj}}{\sigma^2_{cj}} = 0
  \quad\Longrightarrow\quad
  \boxed{\ \hat\mu_{cj} = \frac{1}{N_c}\sum_{i:\,y_i=c} x_{ij}\ }$$

$$\frac{\partial \ell}{\partial \sigma^2_{cj}}
  = \sum_{i:\,y_i=c}\left[ -\frac{1}{2\sigma^2_{cj}}
    + \frac{(x_{ij}-\mu_{cj})^2}{2(\sigma^2_{cj})^2} \right] = 0
  \quad\Longrightarrow\quad
  \boxed{\ \hat\sigma^2_{cj}
    = \frac{1}{N_c}\sum_{i:\,y_i=c}(x_{ij}-\hat\mu_{cj})^2\ }$$

The variance MLE is the **biased** estimator (divide by $N_c$, not
$N_c - 1$) — this is what `sklearn.naive_bayes.GaussianNB` uses, so it is what
we use.

The prior MLE maximises $\sum_c N_c \log \pi_c$ subject to
$\sum_c \pi_c = 1$; a one-line Lagrange multiplier gives the class frequency:

$$\boxed{\ \hat\pi_c = \frac{N_c}{n}\ }$$

A user-supplied `priors` vector overrides this (it must be non-negative,
length $K$, and sum to 1).

**`var_smoothing` — why a variance floor.** A feature that is constant within
a class gives $\hat\sigma^2_{cj} = 0$, and the Gaussian density then divides
by zero. Following scikit-learn, add a floor proportional to the largest
feature variance across the *entire* training set (not per class), which
makes the floor scale-aware:

$$\varepsilon = \texttt{var\_smoothing} \cdot \max_{j}\mathrm{Var}[X_{:,j}],
  \qquad \sigma^2_{cj} \leftarrow \hat\sigma^2_{cj} + \varepsilon$$

Default `var_smoothing` $= 10^{-9}$, matching sklearn. `epsilon_` stores the
resulting $\varepsilon$.

## 4. Predicting — work in log-space

Multiplying $d$ small densities underflows to zero in `float64`, so
prediction is done entirely in logs. Define the **joint log-likelihood** of a
row $x$ with class $c$ (the log of the *unnormalised* numerator of Bayes'
rule):

$$\mathrm{jll}(x, c) = \log\pi_c + \sum_{j=1}^{d}\log\mathcal{N}(x_j;\ \mu_{cj}, \sigma^2_{cj})
  = \log\pi_c - \frac12\sum_{j=1}^{d}\left[\log(2\pi\sigma^2_{cj})
    + \frac{(x_j - \mu_{cj})^2}{\sigma^2_{cj}}\right]$$

- **`predict`** $= \arg\max_c \mathrm{jll}(x, c)$. The evidence
  $\log P(x)$ is the same additive constant for every class in a given row,
  so it does not affect the argmax and never needs to be formed.
- **`predict_proba` / `predict_log_proba`** need that constant. With
  $\mathrm{jll}(x)$ the length-$K$ vector for row $x$,

  $$\log P(c \mid x) = \mathrm{jll}(x, c)
    - \mathrm{logsumexp}_{c'}\mathrm{jll}(x, c'),
    \qquad P(c \mid x) = \exp\big(\log P(c \mid x)\big)$$

  using `scratchgrad.utils.math.logsumexp` (stable, from M0). The subtracted
  term is exactly $\log P(x) = \log\sum_{c'} P(c')P(x \mid c')$.

## 5. Design choices

- **Signature `GaussianNB(*, priors=None, var_smoothing=1e-9)`** — mirrors
  `sklearn.naive_bayes.GaussianNB` (names, defaults, keyword-only). `priors`
  is included because the prior is literally half of Bayes' rule and the
  validation is a few lines; `None` uses the MLE class frequencies.
- **Learned attributes:** `classes_`, `class_count_`, `class_prior_`,
  `mean_` $(K, d)$, `var_` $(K, d)$, `epsilon_`. Readable names per the
  project's notation; sklearn calls `mean_`/`var_` its `theta_`/`var_`, noted
  in the docstring so the reference test's mapping is obvious.
- **Public methods:** `fit`, `predict`, `predict_proba`, `predict_log_proba`
  (one extra line, and log-space *is* the method), `score` (accuracy).
- **Module-level pure functions** factor out the math so tests target it
  directly, mirroring the `_*` helpers in `linear/` and `_vote_proba` in
  `neighbors/`:
  - `_gaussian_log_density(X, mean, var)` → $(n, K, d)$: the per-feature
    $\log\mathcal{N}$ term.
  - `_joint_log_likelihood(X, mean, var, log_prior)` → $(n, K)$: the
    $\mathrm{jll}$ above.
- **No `partial_fit`.** sklearn's online/streaming path is out of scope, the
  same way softmax multiclass was deferred for `LogisticRegression`.
- **Not scale-sensitive.** Each feature gets its own per-class $\mu, \sigma^2$,
  so GaussianNB is equivariant to a per-feature affine rescaling — unlike KNN,
  no standardisation note is needed.

## 6. Algorithm (pseudocode)

```
GaussianNB(priors=None, var_smoothing=1e-9)

fit(X, y):
    if var_smoothing < 0: raise ValueError
    X, y = check_X_y(X, y)
    classes_ = unique(y);  K, d = len(classes_), X.shape[1]
    epsilon_ = var_smoothing * X.var(axis=0).max()          # scale-aware floor
    prior = validate(priors, K)                             # None or (K,) array
    for k, c in enumerate(classes_):
        Xc            = X[y == c]
        class_count_[k] = len(Xc)
        mean_[k]        = Xc.mean(axis=0)                    # μ_cj  (MLE)
        var_[k]         = Xc.var(axis=0)                     # σ²_cj (biased MLE)
    var_ += epsilon_
    class_prior_ = prior if prior is not None else class_count_ / n
    return self

_gaussian_log_density(X, mean, var):                        # -> (n, K, d)
    return -0.5 * (log(2π · var) + (X[:, None, :] - mean)**2 / var)

_joint_log_likelihood(X, mean, var, log_prior):             # -> (n, K)
    return log_prior + _gaussian_log_density(X, mean, var).sum(axis=2)

_jll(X):                                                    # fitted + feature-count checks
    check_is_fitted;  X = check_array(X);  assert X.shape[1] == d
    return _joint_log_likelihood(X, mean_, var_, log(class_prior_))

predict(X):            return classes_[argmax(_jll(X), axis=1)]
predict_log_proba(X):  jll = _jll(X);  return jll - logsumexp(jll, axis=1)[:, None]
predict_proba(X):      return exp(predict_log_proba(X))
score(X, y):           return accuracy_score(y, predict(X))
```

## 7. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-laid tiny 2-class set: `mean_`, `var_` (biased, `/N_c`), `class_prior_`, and one row's `jll` match hand-computed values exactly. |
| Independent recomputation | A pure-Python loop over classes × features rebuilds the log-posterior on seeded random data and agrees with `_joint_log_likelihood` / `predict` / `predict_proba` (the correctness analogue of a gradient check — there is no gradient). |
| `_gaussian_log_density` | Equals `-0.5*log(2*pi*var) - (x-mean)**2/(2*var)` on a hand case; shape `(n, K, d)`. |
| var_smoothing | Feature constant within a class → no div-by-zero, finite `predict_proba`; `epsilon_ == var_smoothing * X.var(0).max()`; larger `var_smoothing` pulls the posterior toward the prior. |
| proba | rows of `predict_proba` sum to 1; shape `(m, K)`; `predict == classes_[argmax(predict_proba)]`; `predict_log_proba ≈ log(predict_proba)`. |
| Multiclass | 3-class `make_blobs`, high held-out accuracy. |
| priors | explicit uniform `priors` differs from frequency priors on imbalanced `y`; a heavily skewed prior flips a borderline prediction; bad length / non-sum-to-1 / negative raises. |
| Contract | `predict` / `predict_proba` before `fit` → `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; feature-count mismatch in `predict` raises; `var_smoothing < 0` raises; `repr` round-trips. |
| Behavioral | Well-separated blobs → near-perfect accuracy; strongly correlated features → argmax still reasonable while `predict_proba` gets overconfident (the naive assumption, documented). |
| Edge | single feature; one sample per class (within-class var 0 → `epsilon_` keeps it finite); binary `y`. |
| Reference (`-m reference`) | `mean_` ↔ `theta_`, `var_`, `class_prior_`, `predict`, and `predict_proba` match `sklearn.naive_bayes.GaussianNB` at `rtol=1e-6`. |

Plus `examples/gaussian_nb.py` — seeded, runnable: fit on Gaussian blobs and
report accuracy and the learned per-class means; then a correlated-features
case showing accuracy holding up while `predict_proba` becomes overconfident.

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd
  ed., §6.6.3 (naive Bayes) and §4.3.
- Bishop, *Pattern Recognition and Machine Learning*, §4.2.
- Murphy, *Machine Learning: A Probabilistic Perspective*, §3.5 and §4.2.
- scikit-learn User Guide, §1.9.1 (Gaussian Naive Bayes).
