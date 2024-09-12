import numpy as np


def syss_dft(
    timestamps: np.array,
    td_samples: np.array,
    dt: float,
    sigmoid_loc: float = 0,
    sigmoid_width: float = 10,
):
    """
    Compute the Fourier transform of discrete data containing
    step-like behaviors as described in [Valencia et al. (arXiv:2406.16636)].
    The Fourier convention follows Eq.(2.16). 


    Parameters
    ----------
    timestamps:
        Time stamps at which the data points were taken.
        Non-zero starting times are properly taken into
        account as phase-shifts [see Eqs. (2.12)-(2.15)].
        Should be equispaced, otherwhise FFT will complain.
    td_samples:
        Data points taken at the specified timestamps.
    dt:
        Timestep used to re-scale the discrete Fourier transform
    sigmoid_loc:
        Position of the auxiliary sigmoid.
        Corresponds to t_jump in Eq.(3.6)
    sigmoid_width:
        Width of the auxiliary sigmoid.
        Corresponds to sigma in Eq.(3.6)

    Returns
    -------
    fd_samples:
        Regularized Discrete Fourier Transform of `td_samples`.
    frequencies:
        Frequencies at which `fd_samples` are evaluated.

    """

    regularizator = fit_sigmoid(td_samples, timestamps, sigmoid_loc, sigmoid_width)

    residual = td_samples - regularizator
    residual_fd = dt * np.fft.fft(residual)

    frequencies = np.fft.fftfreq(len(td_samples), dt)
    # Eq. (3.7)
    fd_samples = residual_fd + sigmoid_fd(
        frequencies, sigmoid_loc - timestamps[0], sigmoid_width
    )  

    return (frequencies, fd_samples)


def fit_sigmoid(
    td_samples: np.array,
    timestamps: np.array,
    sigmoid_loc: float,
    sigmoid_width: float,
):
    """
    Fits the regularizing sigmoid to the discrete dataset according to Eq.(3.6).


    Parameters
    ------------
    td_samples:
        y-coordinates of the data points

    timestamps:
        time-coordinates at which td_samples is evaluated

    sigmoid_loc:
        time coordinate at which the regularizing sigmoid is centered.
        Corresponds to t_{jump} in Eq.(3.6)

    sigmoid_width:
        spreadness of the regularizing sigmoid
        Corresponds to sigma in Eq.(3.6)

    Returns
    ------------
        y-coordinates of the regularizing sigmoid.

    """

    one_zero_sigmoid = sigmoid(timestamps, loc=sigmoid_loc, width=sigmoid_width)

    denominator = one_zero_sigmoid[-1] - one_zero_sigmoid[0]
    fit_amplitude = (td_samples[-1] - td_samples[0]) / denominator
    fit_offset = (
        one_zero_sigmoid[-1] * td_samples[0] - one_zero_sigmoid[0] * td_samples[-1]
    ) / denominator

    return (
        fit_amplitude * sigmoid(timestamps, loc=sigmoid_loc, width=sigmoid_width)
        + fit_offset
    )


def sigmoid(
    t: float,
    loc: float,
    width: float,
):
    """
    Defines a regularizing sigmoid with amplitude 1 and offset 0.


    Parameters
    ----------
    t:
        time-coordinates at which the sigmoid is evaluated.
    loc:
        time coordinate at which the sigmoid is centered.
    width:
        spreadness of the sigmoid.

    Returns
    -------
        y-coordinates of the sigmoid.

    """
    return 0.5 * (np.tanh((t - loc) / width) + 1)


def sigmoid_fd(f: float, loc: float, width: float):
    """
    Closed-form continuous Fourier transform of `sigmoid` 
    following the conventions of Eq.(2.1).


    Parameters
    ----------
    f:
        frequency samples.
    loc:
        time coordinate at which the sigmoid is centered.
    width:
        spreadness of the sigmoid
    Returns
    -------
        y-coordinates of the Fourier transform of `sigmoid`
    """
    return (
        -0.5j
        * np.pi
        * width
        * np.exp(-2j * np.pi * f * loc)
        / np.sinh(np.pi**2 * f * width)
    )
