import numpy as np


def syss_dft(
    timestamps: np.array,
    td_samples: np.array,
    dt: float,
    sigmoid_loc: float = 0,
    sigmoid_width: float = 10,
):
    """
    EXPLAIN HERE THE FT CONVENTIONS.

    Parameters
    ----------
    timestamps:
        
    td_samples:

    dt:
        Timestep. This is used to re-scale the FT.
    sigmoid_loc:
        
    sigmoid_width:
        

    Returns
    -------
    frequencies:
        
    fd_samples:
        Regularized DFT of td_samples 
    """

    regularizator = fit_sigmoid(td_samples, timestamps, sigmoid_loc, sigmoid_width)

    residual = td_samples - regularizator
    residual_fd = dt * np.fft.fft(residual)

    frequencies = np.fft.freq(len(td_samples), dt)
    fd_samples = (
        residual_fd
        + sigmoid_fd(frequencies, sigmoid_loc - timestamps[0], sigmoid_width),
    )

    return (frequencies, fd_samples)


def fit_sigmoid(
    td_samples: np.array,
    timestamps: np.array,
    sigmoid_loc: float,
    sigmoid_width: float,
):

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
    return 0.5 * (np.tanh((t - loc) / width) + 1)


def sigmoid_fd(f: float, loc: float, width: float):
    """
    Closed-form continuous FT of `sigmoid`.
    See `syss_dft` for conventions.
    """
    return (
        -0.5j
        * np.pi
        * width
        * np.exp(-2j * np.pi * f * loc)
        / np.sinh(np.pi**2 * f * width)
    )
