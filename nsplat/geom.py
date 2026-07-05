#!/usr/bin/env python3
"""
Geometry utils adapted from
https://github.com/colormotor/calligraph
"""

import numpy as np
from numpy import sin, cos, tan
from numpy.linalg import norm, det, inv
from scipy.interpolate import interp1d, splprep, splev
import numbers


def is_number(x):
    return isinstance(x, numbers.Number)


def is_compound(S):
    """Returns True if S is a compound polyline,
    a polyline is represented as a list of points, or a numpy array with as many rows as points
    """
    if type(S) != list:
        return False
    if type(S) == list:  # [0])==list:
        if not S:
            return True
        if is_number(S[0][0]):
            return False
        return True
    if type(S[0]) == np.ndarray and len(S[0].shape) > 1:
        return True
    return False


def is_polyline(P):
    """A polyline can be represented as either a list of points or a NxDim array"""
    if type(P[0]) == np.ndarray and len(P[0].shape) < 2:
        return True
    if type(P) == list:
        if is_number(P[0]):
            return False
        else:
            return is_number(P[0][0])
    return False


def close_path(P):
    if type(P) == list:
        return P + [P[0]]
    return np.vstack([P, P[0]])


def close(S):
    if is_compound(S):
        return [close_path(P) for P in S]
    return close_path(S)


def is_empty(S):
    if type(S) == list and not S:
        return True
    return False


def vec(*args):
    return np.array(args)


def colvec(*args):
    return np.array(args).reshape(-1, 1)


def direction(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def radians(x):
    return np.pi / 180 * x


def degrees(x):
    return x * (180.0 / np.pi)


def normalize(v):
    return v / np.linalg.norm(v)


def wedge2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def angle_between(*args):
    """Angle between two vectors (2d) [-pi,pi]"""
    if len(args) == 2:
        a, b = args
    else:
        p1, p2, p3 = args
        a = p3 - p2  # TODO checkme
        b = p1 - p2
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def distance(a, b):
    return norm(b - a)


def distance_sq(a, b):
    return np.dot(b - a, b - a)


def normc(X):
    """Normalizes columns of X"""
    l = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    Y = np.array(X)
    return Y / np.maximum(l, 1e-9).reshape(-1, 1)


def tangents(P, closed=False):
    if closed:
        P = np.vstack([P, P[0]])
    return np.diff(P, axis=0)


def edges(P, closed=False):
    if closed:
        P = close(P)
    return zip(P, P[1:])


def turning_angles(P, closed=False, all_points=False, N=None):
    if len(P) <= 2:
        return np.zeros(len(P))
    if N is None:
        D = tangents(P, closed)
        N = np.array([-perp(normalize(d)) for d in D])

    if closed:
        N = np.vstack([N[-1], N])

    n = len(N)
    A = np.zeros(n - 1)
    for i in range(n - 1):
        A[i] = angle_between(N[i], N[i + 1])

    if all_points and not closed:
        A = np.concatenate([[0], A, [0]])
    return A


def normals_2d(P, closed=0, vertex=False):
    """2d normals (fixme)"""
    # pdb.set_trace()
    if closed:
        # P = np.vstack([P[-1], P, P[0]])
        P = np.vstack([P, P[0]])

    D = np.diff(P, axis=0)
    if vertex and D.shape[0] > 1:
        T = D[:-1] + D[1:]
        if not closed:
            T = np.vstack([T[0], T, T[-1]])
        N = np.dot([[0, 1], [-1, 0]], T.T).T
    else:
        T = D
        N = np.dot([[0, 1], [-1, 0]], T.T).T
        if not closed:
            N = np.vstack([N, N[-1]])
        # else:
        #    N = N[:-1]
    return normc(N)


def point_line_distance(p, a, b):
    if np.array_equal(a, b):
        return distance(p, a)
    else:
        return abs(det(np.array([b - a, a - p]))) / norm(b - a)


def signed_point_line_distance(p, a, b):
    if np.array_equal(a, b):
        return distance(p, a)
    else:
        return (det(np.array([b - a, a - p]))) / norm(b - a)


def point_segment_distance(p, a, b):
    a, b = np.array(a), np.array(b)
    d = b - a
    # relative projection length
    u = np.dot(p - a, d) / np.dot(d, d)
    u = np.clip(u, 0, 1)

    proj = a + u * d
    return np.linalg.norm(proj - p)


def project(p, a, b):
    """Project point p on segment a, b"""
    d = b - a
    t = np.dot(p - a, d) / np.dot(d, d)
    return a + d * t


def perp(x):
    """2d perpendicular vector"""
    return np.dot([[0, -1], [1, 0]], x)


def reflect(a, b):
    d = np.dot(b, a)
    return a - b * d * 2


def line_intersection_uv(a1, a2, b1, b2, aIsSegment=False, bIsSegment=False):
    EPS = 0.00001
    intersection = np.zeros(2)
    uv = np.zeros(2)

    denom = (b2[1] - b1[1]) * (a2[0] - a1[0]) - (b2[0] - b1[0]) * (a2[1] - a1[1])
    numera = (b2[0] - b1[0]) * (a1[1] - b1[1]) - (b2[1] - b1[1]) * (a1[0] - b1[0])
    numerb = (a2[0] - a1[0]) * (a1[1] - b1[1]) - (a2[1] - a1[1]) * (a1[0] - b1[0])

    if abs(denom) < EPS:
        return False, intersection, uv

    uv[0] = numera / denom
    uv[1] = numerb / denom

    intersection[0] = a1[0] + uv[0] * (a2[0] - a1[0])
    intersection[1] = a1[1] + uv[0] * (a2[1] - a1[1])

    isa = True
    if aIsSegment and (uv[0] < 0 or uv[0] > 1):
        isa = False
    isb = True
    if bIsSegment and (uv[1] < 0 or uv[1] > 1):
        isb = False

    res = isa and isb
    return res, intersection, uv


def line_intersection(a1, a2, b1, b2, aIsSegment=False, bIsSegment=False):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, False)
    return res, intersection


def line_segment_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, True)
    return res, intersection


def segment_line_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, True, False)
    return res, intersection


def segment_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, True, True)
    return res, intersection


def line_ray_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, False)
    return res and uv[1] > 0, intersection


def ray_line_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, False)
    return res and uv[0] > 0, intersection


def ray_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, False)
    return res and uv[0] > 0 and uv[1] > 0, intersection


def ray_segment_intersection(a1, a2, b1, b2):
    res, intersection, uv = line_intersection_uv(a1, a2, b1, b2, False, True)
    return res and uv[0] > 0 and uv[1] > 0, intersection


# Rect utilities
def bounding_box(S, padding=0):
    """Axis ligned bounding box of one or more contours (any dimension)
    Returns [min,max] list"""
    if not is_compound(S):
        S = [S]
    if not S:
        return np.array([0, 0]), np.array([0, 0])

    bmin = np.min([np.min(V, axis=0) for V in S if len(V)], axis=0)
    bmax = np.max([np.max(V, axis=0) for V in S if len(V)], axis=0)
    return [bmin - padding, bmax + padding]


bbox = bounding_box


def rect_w(rect):
    return (np.array(rect[1]) - np.array(rect[0]))[0]


def rect_h(rect):
    return (np.array(rect[1]) - np.array(rect[0]))[1]


def rect_size(rect):
    return np.array(rect[1]) - np.array(rect[0])


def rect_aspect(rect):
    return rect_w(rect) / rect_h(rect)


def pad_rect(rect, pad):
    return np.array(rect[0]) + pad, np.array(rect[1]) - pad


def make_rect(x, y, w, h):
    return [np.array([x, y]), np.array([x + w, y + h])]


rect = make_rect


def make_centered_rect(p, size):
    return make_rect(p[0] - size[0] * 0.5, p[1] - size[1] * 0.5, size[0], size[1])


def rect_center(rect):
    return rect[0] + (rect[1] - rect[0]) / 2


def rect_corners(rect, close=False):
    w, h = rect_size(rect)
    rect = (np.array(rect[0]), np.array(rect[1]))
    P = [rect[0], rect[0] + [w, 0], rect[1], rect[0] + [0, h]]
    if close:
        P.append(P[0])
    return P


def rect_l(rect):
    return rect[0][0]


def rect_r(rect):
    return rect[1][0]


def rect_t(rect):
    return rect[0][1]


def rect_b(rect):
    return rect[1][1]


def random_point_in_rect(box):
    x = np.random.uniform(box[0][0], box[1][0])
    y = np.random.uniform(box[0][1], box[1][1])
    return np.array([x, y])


def scale_rect(rect, s, halign=0, valign=0):
    if is_number(s):
        s = [s, s]
    sx, sy = s
    r = [np.array(rect[0]), np.array(rect[1])]
    origin = rect_center(rect)
    if halign == -1:
        origin[0] = rect_l(rect)
    if halign == 1:
        origin[0] = rect_r(rect)
    if valign == -1:
        origin[1] = rect_t(rect)
    if valign == 1:
        origin[1] = rect_b(rect)
    A = trans_2d(origin) @ scaling_2d([sx, sy]) @ trans_2d(-origin)

    return [affine_transform(A, r[0]), affine_transform(A, r[1])]


def rect_in_rect(src, dst, padding=0.0, axis=None):
    """Fit src rect into dst rect, preserving aspect ratio of src, with optional padding"""
    dst = pad_rect(dst, padding)

    dst_w, dst_h = dst[1] - dst[0]
    src_w, src_h = src[1] - src[0]

    ratiow = dst_w / src_w
    ratioh = dst_h / src_h
    if axis == None:
        if ratiow <= ratioh:
            axis = 1
        else:
            axis = 0
    if axis == 1:  # fit vertically [==]
        w = dst_w
        h = src_h * ratiow
        x = dst[0][0]
        y = dst[0][1] + dst_h * 0.5 - h * 0.5
    else:  # fit horizontally [ || ]
        w = src_w * ratioh
        h = dst_h

        y = dst[0][1]
        x = dst[0][0] + dst_w * 0.5 - w * 0.5

    return make_rect(x, y, w, h)


def rect_to_rect_transform(src, dst):
    """Fit src rect into dst rect, without preserving aspect ratio"""
    m = np.eye(3, 3)

    sw, sh = rect_size(src)
    dw, dh = rect_size(dst)

    m = trans_2d([dst[0][0], dst[0][1]])
    m = np.dot(m, scaling_2d([dw / sw, dh / sh]))
    m = np.dot(m, trans_2d([-src[0][0], -src[0][1]]))

    return m


def rect_in_rect_transform(src, dst, padding=0.0, axis=None):
    """Return homogeneous transformation matrix that fits src rect into dst"""
    fitted = rect_in_rect(src, dst, padding, axis)

    cenp_src = rect_center(src)
    cenp_dst = rect_center(fitted)

    M = np.eye(3)
    M = np.dot(M, trans_2d(cenp_dst - cenp_src))
    M = np.dot(M, trans_2d(cenp_src))
    M = np.dot(M, scaling_2d(rect_size(fitted) / rect_size(src)))
    M = np.dot(M, trans_2d(-cenp_src))
    return M


def transform_to_rect(shape, rect, padding=0.0, offset=[0, 0], axis=None):
    """transform a shape or polyline to dest rect"""
    src_rect = bounding_box(shape)
    return affine_transform(
        trans_2d(offset) @ rect_in_rect_transform(src_rect, rect, padding, axis), shape
    )


# 2d transformations (affine)
def det22(mat):
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


def rotate_vector_2d(v, ang):
    """2d rotation matrix"""
    ca = np.cos(ang)
    sa = np.sin(ang)
    x, y = v
    return np.array([x * ca - y * sa, x * sa + y * ca])


def rot_2d(theta, affine=True):
    d = 3 if affine else 2
    m = np.eye(d)
    ct = np.cos(theta)
    st = np.sin(theta)
    m[0, 0] = ct
    m[0, 1] = -st
    m[1, 0] = st
    m[1, 1] = ct

    return m


def trans_2d(xy):
    m = np.eye(3)
    m[0, 2] = xy[0]
    m[1, 2] = xy[1]
    return m


def scaling_2d(xy, affine=True):
    d = 3 if affine else 2

    if is_number(xy):
        xy = [xy, xy]

    m = np.eye(d)
    m[0, 0] = xy[0]
    m[1, 1] = xy[1]
    return m


def shear_2d(xy, affine=True):
    d = 3 if affine else 2
    m = np.eye(d)
    # return m
    m[0, 1] = xy[0]
    m[1, 0] = xy[1]
    return m


t2d = trans_2d
r2d = rot_2d
s2d = scaling_2d
sh2d = shear_2d


# 3d transformations (affine)
def rotx_3d(theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[1, 1] = ct
    m[1, 2] = -st
    m[2, 1] = st
    m[2, 2] = ct

    return m


def roty_3d(theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[0, 0] = ct
    m[0, 2] = st
    m[2, 0] = -st
    m[2, 2] = ct

    return m


def rotz_3d(theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[0, 0] = ct
    m[0, 1] = -st
    m[1, 0] = st
    m[1, 1] = ct

    return m


def trans_3d(xyz):
    m = np.eye(4)
    m[0, 3] = xyz[0]
    m[1, 3] = xyz[1]
    m[2, 3] = xyz[2]
    return m


def scaling_3d(s, affine=True):
    d = 4 if affine else 3
    if not isinstance(s, (list, tuple, np.ndarray)):
        s = [s, s, s]

    m = np.eye(d)
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def _affine_transform_polyline(mat, P):
    dim = P[0].size
    P = np.vstack([np.array(P).T, np.ones(len(P))])
    P = mat @ P
    return list(P[:dim, :].T)


def affine_transform(mat, data):
    if is_empty(data):
        # print('Empty data to affine_transform!')
        return data
    if is_polyline(data):
        P = np.array(data)
        dim = P[0].size
        P = np.vstack([np.array(P).T, np.ones(len(P))])
        P = mat @ P
        return P[:dim, :].T
    elif is_compound(data):
        return [affine_transform(mat, P) for P in data]
    else:  # assume a point
        dim = len(data)
        p = np.concatenate([data, [1]])
        return (mat @ p)[:dim]


def affine_mul(mat, data):
    print("Use affine_transform instead")
    return affine_transform(mat, data)  # For backwards compat


tsm = affine_transform


def projection(mat, data):
    if is_empty(data):
        return data
    if is_polyline(data):
        P = np.array(data)
        dim = P[0].size
        P = np.vstack([P.T, np.ones(len(P))])
        P = mat @ P
        P /= P[-1]
        return P[:dim].T
    elif is_compound(data):
        return [projection(mat, P) for P in data]
    else:
        dim = len(data)
        p = np.concatenate([data, [1]])
        p = mat @ p
        p /= p[-1]
        return p[:dim]


# Generates shapes (as polylines, 2d and 3d)
class shapes:
    def __init__(self):
        pass

    @staticmethod
    def polygon(*args):
        """A closed polygon (joins last point to first)"""
        P = [np.array(p) for p in args]
        P.append(np.array(args[0]))
        return np.array(P)

    @staticmethod
    def regular_polygon(center, r, n):
        ang = ((n - 2) * 180) / (n * 2)
        P = (
            tsm(rot_2d(radians(ang)), shapes.circle([0, 0], r, subd=n, close=False))
            + center
        )
        return P

    @staticmethod
    def circle(center, r, subd=80, unit=1, close=True):
        if subd is None:
            subd = max(3, int(r * 2 * np.pi * (1 / unit)))

        res = np.array(
            [
                vec(np.cos(th), np.sin(th)) * r + center
                for th in np.linspace(0, np.pi * 2, subd + 1)[:-1]
            ]
        )
        if close:
            res = close_path(res)
        return res

    @staticmethod
    def star(radius, ratio_inner=1.0, n=5, center=[0, 0]):
        n = int(max(n, 3))
        th = np.linspace(0, np.pi * 2, n * 2 + 1)[:-1] - np.pi / (n * 2)
        R = [radius, radius / (1.618033988749895 + 1) * ratio_inner]
        P = []
        for i, t in enumerate(th):  # [::-1]):
            r = R[i % 2]
            P.append(direction(t) * r + center)
        P = np.array(P)
        return P

    @staticmethod
    def random_radial_polygon(n, min_r=0.5, max_r=1.0, center=[0, 0]):
        R = np.random.uniform(min_r, max_r, n)
        start = np.random.uniform(0.0, np.pi * 2)
        Theta = randspace(start, start + np.pi * 2, n + 1)
        Theta = Theta[:-1]  # skip last one
        V = np.zeros((n, 2))
        V[:, 0] = np.cos(Theta) * R + center[0]
        V[:, 1] = np.sin(Theta) * R + center[1]
        return V

    @staticmethod
    def rectangle(*args):
        if len(args) == 2:
            rect = [*args]
        elif len(args) == 1:
            rect = args[0]
        elif len(args) == 4:
            rect = make_rect(*args)
        P = np.array(rect_corners(rect))
        return P


def randspace(a, b, n, minstep=0.1, maxstep=0.6):
    """Generate a sequence from a to b with random steps
    minstep and maxstep define the step magnitude
    """
    v = minstep + np.random.uniform(size=(n - 1)) * (maxstep - minstep)
    v = np.hstack([[0.0], v])
    v = v / np.sum(v)
    v = np.cumsum(v)
    return a + v * (b - a)


def curvature(P, closed=0):
    """Contour curvature"""
    P = P.T
    if closed:
        P = np.c_[P[:, -1], P, P[:, 0]]

    D = np.diff(P, axis=1)
    l = (
        np.sqrt(np.sum(np.abs(D) ** 2, axis=0)) + 1e-200
    )  # np.sqrt( D[0,:]**2 + D[1,:]**2 )
    D[0, :] /= l
    D[1, :] /= l

    n = D.shape[1]  # size(D,2);

    theta = np.array([angle_between(a, b) for a, b in zip(D.T, D.T[1:])])

    K = 2.0 * np.sin(theta / 2) / (np.sqrt(l[:-1] * l[1:] + 1e-200))

    if not closed:
        K = np.concatenate([[K[0]], K, [K[-1]]])

    return K


def cleanup_contour(X, eps=1e-10, closed=False, get_inds=False):
    """Removes points that are closer then a threshold eps"""
    if closed:
        X = np.vstack([X, X[0]])
    D = np.diff(X, axis=0)
    inds = np.array(range(X.shape[0])).astype(int)
    # chord lengths
    s = np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2)
    # Delete values in input with zero distance (due to a bug in interp1d)
    I = np.where(s < eps)[0]

    if closed:
        X = X[:-1]
    if len(I):
        X = np.delete(X, I, axis=0)
        inds = np.delete(inds, I)
    if get_inds:
        return X, inds
    return X


def dp_simplify(P, eps, closed=False):
    import cv2

    P = np.array(P)
    dtype = P.dtype
    return cv2.approxPolyDP(P.astype(np.float32), eps, closed).astype(dtype)[:, 0, :]


def uniform_sample(X, delta_s, closed=0, kind="slinear", data=None, eps=1e-10):
    """Uniformly samples a contour at a step dist"""
    if closed:
        X = np.vstack([X, X[0]])
        if data is not None:
            data = np.vstack([data, data[0]])

    D = np.diff(X[:, :2], axis=0)
    # chord lengths
    s = np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2)
    # Delete values in input with zero distance (due to a bug in interp1d)
    I = np.where(s < eps)  # s==0)
    if len(I[0]):
        X = np.delete(X, I, axis=0)
        s = np.delete(s, I)

    if len(X) < 2:
        return X

    if data is not None:
        if type(data) == list or data.ndim == 1:
            data = np.delete(data, I)
        else:
            data = np.delete(data, I, axis=0)

    u = np.cumsum(np.concatenate([[0.0], s]))
    u = u / u[-1]
    n = int(np.ceil(np.sum(s) / delta_s))
    t = np.linspace(u[0], u[-1], n)
    f = interp1d(u, X.T, kind=kind)
    Y = f(t)

    if data is not None:
        f = interp1d(u, data.T, kind=kind)
        data = f(t)
        if closed:
            if data.ndim > 1:
                return Y.T[:-1, :], data.T[:-1, :]
            else:
                return Y.T[:-1, :], data.T[:-1]
        else:
            return Y.T, data.T
    if closed:
        return Y[:, :-1].T
    return Y.T


def subdivide(P, closed=False, n=1):
    if closed:
        P = close(P)
    Q = [P[0]]
    for a, b in zip(P, P[1:]):
        Q.append((a + b) * 0.5)
        Q.append(b)
    if closed:
        Q = Q[:-1]
    if n > 1:
        return subdivide(Q, closed, n - 1)
    return np.array(Q)


# def thick_curve(
#     Xw, scalefn=lambda x: x, smooth_k=0, union=True, add_cap=True, unit=1, **kwargs
# ):
#     from . import clipper

#     Xw = np.array(Xw)
#     Xw[:, 2] = scalefn(Xw[:, 2])
#     try:
#         Ol = curved_offset(Xw[:, :2], Xw[:, 2], smooth_k, **kwargs)[::-1]
#         Or = curved_offset(Xw[:, :2], -Xw[:, 2], smooth_k, **kwargs)
#     except TypeError:
#         # pdb.set_trace()
#         return []

#     stroke = np.vstack([Ol, Or])
#     if union:
#         stroke = clipper.union(stroke, stroke)
#         if add_cap:
#             stroke = clipper.union(
#                 stroke, shapes.circle(Xw[0, :2], Xw[0, 2], subd=None, unit=unit)
#             )
#             stroke = clipper.union(
#                 stroke, shapes.circle(Xw[-1, :2], Xw[-1, 2], subd=None, unit=unit)
#             )

#     return stroke


def curved_offset(spine, widths, n=0, degree=3, closed=False, smooth_k=0):
    # smoothing spline
    if n == 0:
        n = spine.shape[0]
    # print(closed)
    degree = min(degree, n - 1)
    # parameterization
    u = np.linspace(0, 1, spine.shape[0])  # geom.cum_chord_lengths(spine)
    u = u / u[-1]
    spl, u = splprep(
        np.vstack([spine.T, widths]), u=u, k=degree, per=closed, s=smooth_k
    )
    t = np.linspace(0, 1, n)
    x, y, w = splev(t, spl)
    dx, dy, dw = splev(t, spl, der=1)

    centers = np.vstack([x, y])
    tangents = np.vstack([dx, dy]) / (np.sqrt(dx**2 + dy**2) + 1e-10)
    normals = np.vstack([-tangents[1, :], tangents[0, :]])

    pts = (centers + normals * w).T
    if closed:
        pts = np.vstack([pts, pts[0]])

    res = smoothing_spline(n, pts, smooth_k=smooth_k, closed=closed)
    if closed:
        res = np.vstack([res, res[0]])

    return res


def smoothing_spline(
    n,
    pts,
    der=0,
    ds=0.0,
    dim=None,
    closed=False,
    w=None,
    smooth_k=0,
    degree=3,
    alpha=1.0,
):
    """Computes a smoothing B-spline for a sequence of points.
    Input:
    n, number of interpolating points
    pts, sequence of points (dim X m)
    der, derivative order
    ds, if non-zero an approximate arc length parameterisation is used with distance ds between points,
    and the parameter n is ignored.
    closed, if True spline is periodic
    w, optional weights
    smooth_k, smoothing parameter,
    degree, spline degree,
    alpha, parameterisation (1, uniform, 0.5 centripetal)
    """

    if closed:
        pts = np.vstack([pts, pts[0]])

    if w is None:
        w = np.ones(pts.shape[0])
    elif is_number(w):
        w = np.ones(pts.shape[0]) * w

    if dim is None:
        dim = pts.shape[1]
    # D = np.diff(pts, axis=0)
    # # chord lengths
    # s = np.sqrt(np.sum([D[:,i]**2 for i in range(dim)], axis=0))
    # # I = np.where(s==0)
    # # pts = np.delete(pts, I, axis=0)
    # # s = np.delete(s, I)
    # # w = np.delete(w, I)

    # _, I = simplify.cleanup_contour(pts, closed=False, get_indices=True)
    # pts = pts[I]
    # #s = [s[i] for i in I]
    # w = [w[i] for i in I]

    degree = min(degree, pts.shape[0] - 1)

    if pts.shape[0] < 2:
        print("Insufficient points for smoothing spline, returning original")
        return pts

    if ds != 0:
        D = np.diff(pts, axis=0)
        s = np.sqrt(np.sum([D[:, i] ** 2 for i in range(dim)], axis=0)) + 1e-5
        l = np.sum(s)
        s = s ** (alpha)
        u = np.cumsum(np.concatenate([[0.0], s]))
        u = u / u[-1]

        spl, u = splprep(pts.T, w=w, u=u, k=degree, per=closed, s=smooth_k)
        n = max(2, int(l / ds))
        t = np.linspace(u[0], u[-1], n)
    else:
        u = np.linspace(0, 1, pts.shape[0])
        spl, u = splprep(pts.T, u=u, w=w, k=degree, per=closed, s=smooth_k)
        t = np.linspace(0, 1, n)

    if type(der) == list:
        res = []
        for d in der:
            res.append(np.vstack(splev(t, spl, der=d)).T)
        return res
    res = splev(t, spl, der=der)
    return np.vstack(res).T


def chord_lengths(P, closed=0):
    """Chord lengths for each segment of a contour"""
    if closed:
        P = np.vstack([P, P[0]])
    D = np.diff(P, axis=0)
    L = np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2)
    return L


def cum_chord_lengths(P, closed=0):
    """Cumulative chord lengths"""
    if len(P.shape) != 2:
        return []
    if P.shape[0] == 1:
        return np.zeros(1)
    L = chord_lengths(P, closed)
    return np.cumsum(np.concatenate([[0.0], L]))


def chord_length(P, closed=0):
    """Chord length of a contour"""
    if len(P.shape) != 2 or P.shape[0] < 2:
        return 0.0
    L = chord_lengths(P, closed)
    return np.sum(L)


def polygon_area(P):
    if len(P.shape) < 2 or P.shape[0] < 3:
        return 0
    n = P.shape[0]
    area = 0.0
    P = P - np.mean(P, axis=0)
    for i in range(n):
        j = (i + 1) % n
        area += (
            0.5 * (P[i, 1] + P[j, 1]) * (P[i, 0] - P[j, 0])
        )  # trapezoid https://en.wikipedia.org/wiki/Shoelace_formula
        # The triangle version will run into numerical percision errors
        # if we have a small or thin polygon that is quite off center,
        # this can be solved by centering the polygon before comp. Maybe useful to do anyhow?
        # area += 0.5*(P[i,0] * P[j,1] - P[j,0] * P[i,1])

    return area


def polygon_area_tri(P):
    if len(P.shape) < 2 or P.shape[0] < 3:
        return 0
    n = P.shape[0]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += 0.5 * (
            P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
        )  # Seems to be less precise for small numbers?
    return area


def triangle_area(a, b, c):
    da = a - b
    db = c - b
    return det(np.vstack([da, db])) * 0.5


def collinear(a, b, p, eps=1e-5):
    return abs(triangle_area(a, b, p)) < eps


def segments_collinear(a, b, c, d, eps=1e-5):
    return collinear(a, b, c, eps) and collinear(a, b, d, eps)


def left_of(p, a, b, eps=1e-10):
    # Assumes coordinate system with y up so actually will be "right of" if visualizd y down
    p, a, b = [np.array(v) for v in [p, a, b]]
    return triangle_area(a, b, p) < eps


def is_point_in_triangle(p, tri, eps=1e-10):
    L = [left_of(p, tri[i], tri[(i + 1) % 3], eps) for i in range(3)]
    return L[0] == L[1] and L[1] == L[2]


def is_point_in_rect(p, rect):
    """return wether a point is in a rect"""
    l, t = rect[0]
    r, b = rect[1]
    w, h = rect[1] - rect[0]

    return p[0] >= l and p[1] >= t and p[0] <= r and p[1] <= b


def is_point_in_poly(p, P):
    """Return true if point in polygon"""
    c = False
    n = P.shape[0]
    j = n - 1
    for i in range(n):
        if ((P[i, 1] > p[1]) != (P[j, 1] > p[1])) and (
            p[0]
            < (P[j, 0] - P[i, 0]) * (p[1] - P[i, 1]) / (P[j, 1] - P[i, 1]) + P[i, 0]
        ):
            c = not c
        j = i
    return c


def is_point_in_shape(p, S, get_flags=False):
    """Even odd point in shape"""
    if p is None:
        return False
    c = 0
    flags = []
    for P in S:
        if len(P) < 3:
            flags.append(False)
        elif is_point_in_poly(p, P):
            flags.append(True)
            c = c + 1
        else:
            flags.append(False)

    res = (c % 2) == 1
    if get_flags:
        return res, flags
    return res


def select_convex_vertex(P, area=None):
    """Select convex vertex (with arbitrary winding)"""
    if area is None:
        area = polygon_area(P)
    n = len(P)
    maxh = 0
    verts = []
    for v in range(n):
        a, b = (v - 1) % n, (v + 1) % n
        if angle_between(P[v] - P[a], P[b] - P[v]) * area > 0:
            h = point_line_distance(P[v], P[a], P[b])
            if h > maxh:
                maxh = h
                verts.append(v)
    if not verts:
        return None
    return verts[-1]


def get_point_in_polygon(P, area=None):
    """Get a point inside polygon P
    if P is a tuple (S, i), P=S[i] and test considers multiple shape contours
    See http://apodeline.free.fr/FAQ/CGAFAQ/CGAFAQ-3.html
    and O'Rourke"""
    if type(P) == tuple:
        # With a tuple assume we are indexing a shape
        S, ind = P
        P = S[ind]
        test_shape = True
    else:
        S = []

    n = len(P)
    if area is None:
        area = polygon_area(P)
    v = select_convex_vertex(P, area)
    if v is None:
        # from polygonsoup import plut
        # import pdb
        # pdb.set_trace()
        # print('Could not find convex vertex for area %f'%area)
        return None  # np.mean(P, axis=0)

    a, b = (v - 1) % n, (v + 1) % n
    inside = []
    dist = np.inf
    # Check if no other point is inside the triangle a, v, b
    # and select closest to v if any present
    for i in range(n - 3):
        q = (b + 1 + i) % n
        if is_point_in_triangle(P[q], [P[a], P[v], P[b]]):
            d = distance(P[q], P[v])
            if d < dist:
                dist = d
                inside.append(P[q])
    if S:  # With a compound shape we need to also test the other vertices
        for si, Q in enumerate(S):
            if si == ind:
                continue
            n = len(Q)
            for i in range(n):
                if is_point_in_triangle(Q[i], [P[a], P[v], P[b]]):
                    d = distance(Q[i], P[v])  # d = distance(P[q], P[v])
                    if d < dist:
                        dist = d
                        inside.append(Q[i])
    if not inside:
        return (P[a] + P[v] + P[b]) / 3
    # no points inside triangle, select midpoint
    return (inside[-1] + P[v]) / 2


def get_holes(S, get_points_and_areas=False, verbose=False):
    """Return an array with same size as S with 0 not a hole an 1 a hole
    Optionally return positions in sub-contours and their areas"""
    areas = [polygon_area(P) for P in S]
    if verbose:
        from tqdm import tqdm
    else:
        tqdm = lambda v: list(v)

    points = [get_point_in_polygon((S, i), area) for i, area in enumerate(tqdm(areas))]
    holes = [
        True if not is_point_in_shape(points[i], S) else False
        for i, P in enumerate(tqdm(S))
    ]
    if get_points_and_areas:
        return holes, points, areas
    return holes


def get_points_in_holes(S):
    """Get positions inside the holes of S (if any)"""
    holes, points, areas = get_holes(S, get_points_and_areas=True)
    return [points[i] for i, hole in enumerate(holes) if hole]


def fix_shape_winding(S):
    """Fixes shape winding to be consistent:
    for y-down: cw out, ccw in
    for y-up: ccw out, cw in"""
    is_shape = True
    if type(S) != list:
        S = [S]
        is_shape = False
    # Make sure that contours don't have repeated end-points because that would break
    # subsequent computations
    S = [cleanup_contour(P, closed=True) for P in S]
    # Identify holes
    holes, points, areas = get_holes(S, get_points_and_areas=True)
    S2 = []

    for i, P in enumerate(S):
        P = np.array(P)
        if abs(areas[i]) < 1e-10:
            print("zero area sub-shape")
            continue
        if (areas[i] < 0) != holes[i]:
            P = P[::-1]
        S2.append(P)

    if not is_shape:
        return S2[0]
    return S2


def get_polygons_with_holes(S):
    """Return an array with same size as S with 0 not a hole an 1 a hole
    Optionally return positions in sub-contours and their areas"""
    import pdb

    areas = [polygon_area(P) for P in S]
    points = [get_point_in_polygon(P, area) for P, area in zip(S, areas)]
    holes = []
    points_in_flags = []
    for i, P in enumerate(S):
        res, flags = is_point_in_shape(points[i], S, get_flags=True)
        holes.append(not res)
        points_in_flags.append(flags)

    n = len(S)
    polyholes = []

    for i in range(n):
        if holes[i]:
            continue
        P = S[i]
        pholes = []
        for j in range(n):
            if i == j or not holes[j]:
                continue
            if points_in_flags[j][i]:
                pholes.append(S[j])
        polyholes.append((P, pholes))
    return polyholes
