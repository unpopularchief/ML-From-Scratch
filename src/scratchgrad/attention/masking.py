"""Boolean attention masks: ``True`` means the query may attend to the key.

Full derivation: docs/derivations/attention.md section 3.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def causal_mask(t_q: int, t_k: int) -> npt.NDArray[np.bool_]:
    """Lower-triangular ``(t_q, t_k)`` mask: query ``i`` sees keys ``j <= i``."""
    return np.tril(np.ones((t_q, t_k), dtype=bool))


def padding_mask(lengths: npt.ArrayLike, t_k: int) -> npt.NDArray[np.bool_]:
    """``(B, 1, t_k)`` mask that blocks key positions at or beyond each length.

    ``lengths[b]`` is the number of real (non-pad) tokens in sequence ``b``.
    """
    lengths = np.asarray(lengths)
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got shape {lengths.shape}")
    if (lengths < 0).any() or (lengths > t_k).any():
        raise ValueError(f"lengths must lie in [0, {t_k}]")
    return (np.arange(t_k)[None, :] < lengths[:, None])[:, None, :]
