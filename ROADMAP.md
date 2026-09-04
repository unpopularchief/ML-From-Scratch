# Roadmap

Progress tracker for `plan.md` §7. Milestones happen in order — see
`plan.md` §0 for why nothing here is worked on out of sequence.

## M0 — Foundation (`v0.0.1`)

- [x] Repo scaffold (`.gitattributes`, `.gitignore`, `LICENSE`, `pyproject.toml`)
- [x] `base.Estimator`, `typing`, `exceptions`
- [x] `utils.validation` (`check_X_y`, `check_array`, `check_is_fitted`, `check_random_state`)
- [x] `utils.math` (`sigmoid`, `softmax`, `logsumexp`)
- [x] `metrics` (regression, classification, pairwise)
- [x] `preprocessing` (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `train_test_split`)
- [x] `datasets` synthetic generators (`make_regression`, `make_blobs`, `make_moons`)
- [x] `tests/helpers/gradcheck.py` — the finite-difference gradient checker
- [x] CI (`ci.yml`), pre-commit, PR template
- [x] `README.md`, `docs/conventions.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
- [x] Green CI on a pushed commit (`97526d8`, `e01ebf3` — both green on ubuntu + windows)

No algorithms in this milestone — see `plan.md`'s M0 deliverables section.

## M1 — Classical supervised (`v0.1.0`)

- [x] LinearRegression (normal equation + gradient descent)
- [x] Ridge (regularised normal equation + gradient descent)
- [ ] Lasso (coordinate descent)
- [ ] LogisticRegression
- [ ] KNN
- [ ] GaussianNB
- [ ] DecisionTree (CART: gini/entropy/MSE)
- [x] `[reference]` extra + `reference.yml` added (scikit-learn parity tests)

**MVP** = M0 + LinearRegression + LogisticRegression + KNN + one example + green CI.

## M2 — Ensembles & unsupervised (`v0.2.0`)

- [ ] RandomForest
- [ ] AdaBoost
- [ ] GradientBoosting
- [ ] KMeans
- [ ] DBSCAN
- [ ] GaussianMixture (EM)
- [ ] PCA
- [ ] LinearSVM (kernel SVM/SMO optional)

## M3 — Optimizers + MLP, manual backward (`v0.3.0`)

- [ ] `optim`: SGD, momentum, Nesterov, RMSprop, Adam
- [ ] `nn.Module`, `Linear`, activations, losses, init schemes
- [ ] Dropout, BatchNorm
- [ ] Full gradient-check coverage
- [ ] MNIST example
- [ ] PyTorch added to `[reference]`

## M4 — CNN & RNN, still manual backward (`v0.4.0`)

- [ ] Conv2d (im2col), MaxPool, Flatten
- [ ] RNN cell, LSTM, BPTT
- [ ] MNIST CNN example, char-level RNN example

## M5 — Autograd engine (`v0.5.0`)

- [ ] `Tensor`, topological `backward()`, ops + VJPs
- [ ] Tensor-based layers and optimizers
- [ ] Proof test: M3's MLP reimplemented on autograd, gradients match the hand-derived ones

## M6 — Attention (`v0.6.0`)

- [ ] Scaled dot-product attention, causal + padding masks
- [ ] Multi-head attention, positional encodings

## M7 — Transformer, capstone (`v1.0.0`)

- [ ] Pre-LN block, encoder, decoder
- [ ] Char-level tokenizer, then minimal BPE
- [ ] Tiny GPT trained on tiny-shakespeare + write-up
