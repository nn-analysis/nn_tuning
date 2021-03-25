import math
# from numba import cuda, guvectorize, float32


# @guvectorize([(float32[:, :], float32[:], float32[:])], '(m, n), (n)->(n)', target='cuda')
# @cuda.jit(device=True)
def __sum_product__(a, b, result):
    for i in range(0, a.shape[0]):
        row_result = 0
        for j in range(0, a.shape[1]):
            row_result = row_result + a[i, j] * b[j]
        result[i] = row_result


# @guvectorize([(float32[:], float32[:], float32, float32, float32, float32[:, :], float32[:], float32[:])],
#              '(n), (n), (), (), (), (n,n), (n)->(n)', target='cuda')
def __calculate_prediction__(stim_x, stim_y, x, y, sigma, stimulus, g, result):
    g3 = sigma ** 2
    g3 = -2 * g3
    for i in range(0, g.shape[0]):
        g[i] = math.exp(((stim_x[i] - x) ** 2 + (stim_y[i] - y) ** 2) / g3)
    # Calculate prediction
    __sum_product__(stimulus, g, result)


# @guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(a, n), (n, m)->(a, m)', target='cuda')
def __matmultiply__(x, y, result):
    for i in range(0, x.shape[0]):
        for j in range(0, y.shape[1]):
            r = 0
            for k in range(0, y.shape[0]):
                r += x[i, k] * y[j, k]
            result[i, j] = r


# @guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(a, b), (n, m)->(a, m)', target='cuda')
def __multiply__(x, y, result):
    for i in range(0, x.shape[0]):
        for j in range(0, y.shape[1]):
            result[i, j] = x[i, 0] * y[i, j]


# @guvectorize([(float32[:, :], float32[:, :])],
#              '(n, m)->(m, n)', target='cuda')
def __transpose2d__(a, result):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[j, i] = a[i, j]


# @guvectorize([(float32[:, :], float32[:])],
#              '(m)->()', target='cuda')
# @cuda.jit(device=True)
def __mean__(a):
    result = 0
    for i in range(a.shape[0]):
        result += a[i] / a.shape[0]
    return result


# @guvectorize([(float32[:, :], float32[:, :])],
#              '(m)->()', target='cuda')
# @cuda.jit(device=True)
def __var__(a):
    mean = __mean__(a)
    result = 0
    for i in range(a.shape[0]):
        result += math.pow(a[i]-mean, 2) / a.shape[0]
    return result


# @guvectorize([(float32[:, :], float32[:])],
#              '(d, m)->(d)', target='cuda')
def __two_d_var__(a, result):
    for i in range(a.shape[0]):
        result[i] = __var__(a[i])