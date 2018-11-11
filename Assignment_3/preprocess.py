import numpy as np

def convert_to_YIQ(img):
    """RGB to YIQ matrix
    courtesy of http://www.cs.rit.edu/~ncs/color/t_convert.html"""
    assert 0 <= np.max(img) <= 1
    m = np.array([[ 0.299,  0.587,  0.114],
                  [ 0.596, -0.275, -0.321],
                  [ 0.212, -0.523,  0.311]])
    return np.einsum('ij,klj->kli', m, img)


def convert_to_RGB(img):
    """YIQ to RGB matrix
    courtesy of http://www.cs.rit.edu/~ncs/color/t_convert.html"""
    m = np.array([[ 1.,  0.956,  0.621],
                  [ 1., -0.272, -0.647],
                  [ 1., -1.105,  1.702]])
    return np.einsum('ij,klj->kli', m, img)


def remap_luminance(A, Ap, B):
    # single channel only
    assert(len(A.shape) == len(Ap.shape) == len(B.shape) == 2)

    m_A = np.mean(A)
    m_B = np.mean(B)
    s_A = np.std(A)
    s_B = np.std(B)

    A_remap = (s_B/s_A) * ( A - m_A) + m_B
    Ap_remap = (s_B/s_A) * ( Ap - m_A) + m_B

    return A_remap, Ap_remap
