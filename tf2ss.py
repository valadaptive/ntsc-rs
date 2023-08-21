from numpy import (r_, eye, atleast_2d,
                   asarray, zeros, array, outer, ndarray)
import numpy as np

import math

from scipy.signal import butter, lfilter, lfiltic, dimpulse, dlti, lfilter_zi
from scipy import linalg

def tf2ss(num, den):
    r"""Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the coefficients of the numerator and
        denominator polynomials, in order of descending degree. The
        denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    Examples
    --------
    Convert the transfer function:

    .. math:: H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    to the state-space representation:

    .. math::

        \dot{\textbf{x}}(t) =
        \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t) \\

        \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)

    >>> from scipy.signal import tf2ss
    >>> A, B, C, D = tf2ss(num, den)
    >>> A
    array([[-2., -1.],
           [ 1.,  0.]])
    >>> B
    array([[ 1.],
           [ 0.]])
    >>> C
    array([[ 1.,  2.]])
    >>> D
    array([[ 1.]])
    """
    # Controller canonical state-space representation.
    #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
    #  states are found by asserting that X(s) = U(s) / D(s)
    #  then Y(s) = N(s) * X(s)
    #
    #   A, B, C, and D follow quite naturally.
    nn = len(num.shape)
    if nn == 1:
        num = asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return (array([], float), array([], float), array([], float),
                array([], float))

    # pad numerator to have same number of columns has denominator
    num = r_['-1', zeros((num.shape[0], K - M), num.dtype), num]

    print("num:", num, num.shape)

    if num.shape[-1] > 0:
        D = atleast_2d(num[:, 0])
        print("D:", D)

    else:
        # We don't assign it an empty array because this system
        # is not 'null'. It just doesn't have a non-zero D
        # matrix. Thus, it should have a non-zero shape so that
        # it can be operated on by functions like 'ss2tf'
        D = array([[0]], float)

    if K == 1:
        D = D.reshape(num.shape)

        return (zeros((1, 1)), zeros((1, D.shape[1])),
                zeros((D.shape[0], 1)), D)

    frow = -array([den[1:]])
    A = r_[frow, eye(K - 2, K - 1)]
    B = eye(K - 1, 1)

    print("num den trimmed:", num[:, 0], den[1:])
    print("num 0:", num[0][0])
    print("sub:", num[0][0] * den[1:])

    C = num[:, 1:] - outer(num[:, 0], den[1:])
    print("alt C:", num[0][1:] - (num[0][0] * den[1:]), ", C:", C)
    D = D.reshape((C.shape[0], B.shape[1]))

    return A, B, C, D

def lfilter_zi_alt(b, a):
    """
    Construct initial conditions for lfilter for step response steady-state.

    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.

    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.

    See Also
    --------
    lfilter, lfiltic, filtfilt

    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::

        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)

    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::

        zi = A*zi + B

    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.

    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::

        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]

    assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
    divided by a[0].

    Examples
    --------
    The following code creates a lowpass Butterworth filter. Then it
    applies that filter to an array whose values are all 1.0; the
    output is also all 1.0, as expected for a lowpass filter.  If the
    `zi` argument of `lfilter` had not been given, the output would have
    shown the transient signal.

    >>> from numpy import array, ones
    >>> from scipy.signal import lfilter, lfilter_zi, butter
    >>> b, a = butter(5, 0.25)
    >>> zi = lfilter_zi(b, a)
    >>> y, zo = lfilter(b, a, ones(10), zi=zi)
    >>> y
    array([1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

    Another example:

    >>> x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    >>> y, zf = lfilter(b, a, x, zi=zi*x[0])
    >>> y
    array([ 0.5       ,  0.5       ,  0.5       ,  0.49836039,  0.48610528,
        0.44399389,  0.35505241])

    Note that the `zi` argument to `lfilter` was computed using
    `lfilter_zi` and scaled by `x[0]`.  Then the output `y` has no
    transient until the input drops from 0.5 to 0.0.

    """

    # FIXME: Can this function be replaced with an appropriate
    # use of lfiltic?  For example, when b,a = butter(N,Wn),
    #    lfiltic(b, a, y=numpy.ones_like(a), x=numpy.ones_like(b)).
    #

    # We could use scipy.signal.normalize, but it uses warnings in
    # cases where a ValueError is more appropriate, and it allows
    # b to be 2D.
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")
    a = np.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")

    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    if a[0] != 1.0:
        # Normalize the coefficients so a[0] == 1.
        b = b / a[0]
        a = a / a[0]

    n = max(len(a), len(b))

    # Pad a or b with zeros so they are the same length.
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a), dtype=a.dtype)]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b), dtype=b.dtype)]

    IminusA = np.eye(n - 1, dtype=np.result_type(a, b)) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]

    print("num:", b)
    print("den:", a)

    '''print("a:", a)
    print("eye:", np.eye(n - 1, dtype=np.result_type(a, b)))
    print("companion:", linalg.companion(a).T)
    print("IminusA:", IminusA)
    print("IminusA sum:", IminusA[:,0].sum())
    print("a sum:", a.sum())'''
    print("B sum:", B.sum())
    print("B:", B)
    print(a[1:] * b[0])

    # print("IminusA:", IminusA[:,0], a)
    # print("B:", B)
    # Solve zi = A*zi + B
    #zi = np.linalg.solve(IminusA, B)

    # For future reference: we could also use the following
    # explicit formulas to solve the linear system:
    #
    zi = np.zeros(n - 1)
    zi[0] = B.sum() / a.sum()
    asum = 1.0
    csum = 0.0
    for k in range(1,n-1):
        asum += a[k]
        csum += b[k] - a[k]*b[0]
        zi[k] = asum*zi[0] - csum

    return zi

filt = (
    array([0.5]),
    array([2.3, -1.2])
)

#filt = butter(3, 0.5)
print(filt)

data = [4, 1, 2, 0, 0, 0, 0]


def lowpassFilter(cutoff: float, reset: float, rate: float = 315000000.00 / 88 * 4):
    timeInterval = 1.0 / rate
    tau = 1 / (cutoff * 2.0 * math.pi)
    alpha = timeInterval / (tau + timeInterval)

    return array([alpha]), array([1, -(1.0 - alpha)])

# filter = lowpassFilter(600000.0, 0.0)
filter = butter(5, 0.5)
# filter = (filter[0], filter[1][1:])
# filter = (filter[0][2:], filter[1])
print("filter tf:", filter)
# state = tf2ss(filter[0], filter[1])
# print(state)

# ic = lfiltic(filter[0], filter[1], y=[16.0], x=lfilter_zi(filter[0], filter[1]))
ic = lfilter_zi(filter[0], filter[1])
ic_alt = lfilter_zi_alt(filter[0], filter[1])
# ic_better = lfiltic(filter[0], filter[1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1])
filtered = lfilter(filter[0], filter[1], data, zi=ic * 4.0)
# filtered2 = lfilter(filter[0], filter[1], data, zi=ic_better)
print("filtered:", filtered)
# print("filtered2:", filtered2)
print("ic:", ic)
print("ic (ours):", ic_alt)
# print("ic (better):", ic_better)

'''impulse_data = lfilter(filter[0], filter[1], [1, 0, 0, 0, 0, 0, 0, 0, 0])
print("lfilter impulse:", impulse_data)

alt_impulse = dimpulse(dlti(filter[0], filter[1]), n = 9)
print("dimpulse impulse:", alt_impulse[1])'''