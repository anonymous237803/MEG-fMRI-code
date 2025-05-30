import numpy as np
import scipy.stats
import warnings
from scipy.stats import gamma
import torch
from torch import nn
import torch.fft
import pickle
from scipy.stats import zscore


def get_hrf(shape="twogamma", tr=1, pttp=5, nttp=15, pos_neg_ratio=6, onset=0, pdsp=1, ndsp=1, t=None):
    """Create canonical hemodynamic response filter

    Parameters
    ----------
    shape : string, {'twogamma'|'boynton'}
        HRF general shape {'twogamma' [, 'boynton']}
    tr : scalar
        HRF sample frequency, in seconds (default = 2)
    pttp : scalar
        time to positive (response) peak in seconds (default = 5)
    nttp : scalar
        Time to negative (undershoot) peak in seconds (default = 15)
    pos_neg_ratio : scalar
        Positive-to-negative ratio (default: 6, OK: [1 .. Inf])
    onset :
        Onset of the HRF (default: 0 secs, OK: [-5 .. 5])
    pdsp :
        Dispersion of positive gamma PDF (default: 1)
    ndsp :
        Dispersion of negative gamma PDF (default: 1)
    t : vector | None
        Sampling range (default: [0, onset + 2 * (nttp + 1)])

    Returns
    -------
    h : HRF function given within [0 .. onset + 2*nttp]
    t : HRF sample points

    Notes
    -----
    The pttp and nttp parameters are increased by 1 before given
    as parameters into the scipy.stats.gamma.pdf function (which is a property
    of the gamma PDF!)

    Based on hrf function in matlab toolbox `BVQXtools`; converted to python and simplified by ML
    Version:  v0.7f
    Build:    8110521
    Date:     Nov-05 2008, 9:00 PM CET
    Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
    URL/Info: http://wiki.brainvoyager.com/BVQXtools
    """

    # Input checks
    if not shape.lower() in ("twogamma", "boynton"):
        warnings.warn('Shape can only be "twogamma" or "boynton"')
        shape = "twogamma"
    if t is None:
        t = np.arange(0, (onset + 2 * (nttp + 1)), tr) - onset
    else:
        t = np.arange(np.min(t), np.max(t), tr) - onset

    # Create filter
    h = np.zeros((len(t),))
    if shape.lower() == "boynton":
        # boynton (single-gamma) HRF
        h = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
    elif shape.lower() == "twogamma":
        gpos = gamma.pdf(t, pttp + 1, pdsp)
        gneg = gamma.pdf(t, nttp + 1, ndsp) / pos_neg_ratio
        h = gpos - gneg
    h /= np.sum(h)
    return t, h


def zscore_tensor(tensor, dim=0):
    """Z-score a tensor along a given dimension"""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, correction=0)
    tensor_norm = torch.where(std > 0, (tensor - mean) / std, torch.zeros_like(tensor))
    return tensor_norm


def flatten_recursive(lst):
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(flatten_recursive(x))
        else:
            result.append(x)
    return result


def hilbert_torch(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Analytic signal via Hilbert transform, PyTorch version.

    Parameters
    ----------
    x   : real-valued tensor (..., N)
    dim : time axis

    Returns
    -------
    z   : complex tensor with same shape as x
    """
    N = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)

    # Build frequency-domain multiplier H
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    if N % 2 == 0:  # even length
        h[0] = h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:  # odd length
        h[0] = 1.0
        h[1 : (N + 1) // 2] = 2.0

    # Reshape h so it can broadcast along all leading dims
    shape = [1] * x.ndim
    shape[dim] = N
    h = h.reshape(shape)

    Zf = Xf * h  # apply half-filter
    z = torch.fft.ifft(Zf, dim=dim)  # complex analytic signal
    return z


class CorrelationLoss(nn.Module):
    def __init__(self, dim: int = 0, reduction: str = "mean"):
        """
        Correlation loss = 1 - Pearson correlation coefficient along `dim`.

        Args:
            dim: the axis along which to compute correlation.
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred and y_true must have the same shape, got {y_pred.shape} vs {y_true.shape}")

        # z-score along the specified dimension
        y_pred_z = zscore_tensor(y_pred, dim=self.dim)
        y_true_z = zscore_tensor(y_true, dim=self.dim)

        # Pearson r is the mean of elementwise product along that axis
        corr = (y_pred_z * y_true_z).mean(dim=self.dim)

        # loss = 1 - r
        loss = 1 - corr

        if self.reduction == "mean":
            if weight is not None:
                # Assume weight is 1D matching the last dimension of loss
                if weight.shape != (loss.shape[-1],):
                    raise ValueError(f"Weight must have shape {(loss.shape[-1],)}, but got {weight.shape}")
                # Expand weight to match loss shape
                while weight.dim() < loss.dim():
                    weight = weight.unsqueeze(0)
                return (loss * weight).sum() / weight.sum()
            else:
                return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def get_mean_rate(SUBJECT):

    # story to session and block mapping
    with open("data/story_sess_block.pkl", "rb") as f:
        story_sess_block = pickle.load(f)
    all_stories = list(story_sess_block.keys())

    rates = []
    for this_story in all_stories:
        # get session and block
        session, block = story_sess_block[this_story]
        # get the delay correction coefficients
        with open(f"/data/story_dataset/moth_meg/{session}/delay_correction/coefficients/{SUBJECT}/{this_story}.pkl", "rb") as f:
            delay_correction_coefs = pickle.load(f)
        rate = delay_correction_coefs["rate"]
        rates.append(rate)
    mean_rate = np.mean(rates)

    return mean_rate


def get_stretched_features(time_features, SUBJECT, session, this_story, use_mean_rate=False):

    if use_mean_rate == True:

        # get mean rate
        mean_rate = get_mean_rate(SUBJECT)

        # correct the phoneme times
        time_features_corrected = [tuple([t[0] * mean_rate, t[1] * mean_rate] + list(t[2:])) for t in time_features]

    else:

        # get the delay correction coefficients
        with open(f"/data/story_dataset/moth_meg/{session}/delay_correction/coefficients/{SUBJECT}/{this_story}.pkl", "rb") as f:
            delay_correction_coefs = pickle.load(f)
        rate = delay_correction_coefs["rate"]

        # correct the phoneme times
        time_features_corrected = [tuple([t[0] * mean_rate, t[1] * mean_rate] + list(t[2:])) for t in time_features]

    return time_features_corrected

# some voxels have 0 std
def nan_zscore(data, axis=0):
    data = zscore(data, axis=axis)
    data[np.isnan(data)] = 0
    return data


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, best_meg_loss, best_fmri_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "sched_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_meg_loss": best_meg_loss,
        "best_fmri_loss": best_fmri_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["opt_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["sched_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"], checkpoint["best_meg_loss"], checkpoint["best_fmri_loss"]


def fdr_correction(p_values, alpha=0.05, method="bh", axis=None):
    """
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Modified from the code at https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    Args:
        p_values: The p_values to correct.
        alpha: The error rate to correct the p-values with.
        method: one of by (for Benjamini/Yekutieli) or bh for Benjamini/Hochberg
        axis: Which axis of p_values to apply the correction along. If None, p_values is flattened.
    Returns:
        indicator_alternative: An boolean array with the same shape as p_values_corrected that is True where
            the null hypothesis should be rejected
        p_values_corrected: The p_values corrected for FDR. Same shape as p_values
    """
    p_values = np.asarray(p_values)

    shape = p_values.shape
    if axis is None:
        p_values = np.reshape(p_values, -1)
        axis = 0
    if axis < 0:
        axis += len(p_values.shape)
        if axis < 0:
            raise ValueError("axis out of bounds")

    indices_sorted = np.argsort(p_values, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)

    correction_factor = np.arange(1, p_values.shape[axis] + 1) / p_values.shape[axis]
    correction_factor_shape = [1] * len(p_values.shape)
    correction_factor_shape[axis] = len(correction_factor)
    correction_factor = np.reshape(correction_factor, correction_factor_shape)

    if method == "bh":
        pass
    elif method == "by":
        c_m = np.sum(1 / np.arange(1, p_values.shape[axis] + 1))
        correction_factor = correction_factor / c_m
    else:
        raise ValueError("Unrecognized method: {}".format(method))

    # set everything left of the maximum qualifying p-value
    indicator_alternative = p_values <= correction_factor * alpha
    indices_all = np.reshape(
        np.arange(indicator_alternative.shape[axis]),
        (1,) * axis + (indicator_alternative.shape[axis],) + (1,) * (len(indicator_alternative.shape) - 1 - axis),
    )
    indices_max = np.nanmax(np.where(indicator_alternative, indices_all, np.nan), axis=axis, keepdims=True).astype(int)
    indicator_alternative = indices_all <= indices_max
    del indices_all

    p_values = np.clip(
        np.take(
            np.minimum.accumulate(
                np.take(
                    p_values / correction_factor,
                    np.arange(p_values.shape[axis] - 1, -1, -1),
                    axis=axis,
                ),
                axis=axis,
            ),
            np.arange(p_values.shape[axis] - 1, -1, -1),
            axis=axis,
        ),
        a_min=0,
        a_max=1,
    )

    indices_sorted = np.argsort(indices_sorted, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)
    indicator_alternative = np.take_along_axis(indicator_alternative, indices_sorted, axis=axis)

    return np.reshape(indicator_alternative, shape), np.reshape(p_values, shape)


def corr_multi_torch(y_true, y_pred, dim=0):
    """
    Compute correlation between y_true and y_pred per target dimension.
    Inputs: (n_samples, n_targets)
    """
    y_true_z = zscore_tensor(y_true, dim=dim)
    y_pred_z = zscore_tensor(y_pred, dim=dim)

    return torch.mean(y_true_z * y_pred_z, dim=dim)