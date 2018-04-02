import numpy as np
from pygeom import rectangle as rt
from pygeom import pixel_rect as pr

def create(ps, dtype=None): return _C(ps).astype(dtype)

def aspoly(ps): return _C(ps)

def npoint(ps): return _B(ps).shape[1] >> 1

def bounding_rect(ps): 
	ps = _B(ps).reshape(len(ps), -1 ,2)
	xs, ys = ps[:, :, 0], ps[:, :, 1]
	min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)
	min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)
	return rt.create(min_x, min_y, max_x-min_x, max_y-min_y)

def bounding_pixel_rect(ps): 
	ps = _B(ps).reshape(len(ps), -1 ,2)
	xs, ys = ps[:, :, 0], ps[:, :, 1]
	min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)
	min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)
	return pr.create(min_x, min_y, max_x-min_x, max_y-min_y)

def _npoint(ps): return ps.shape[1] >> 1

def _bounding_rect(ps): 
	ps = ps.reshape(len(ps), -1 ,2)
	xs, ys = ps[:, :, 0], ps[:, :, 1]
	min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)
	min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)
	return rt.create(min_x, min_y, max_x-min_x, max_y-min_y)

def _bounding_pixel_rect(ps): 
	ps = ps.reshape(len(ps), -1 ,2)
	xs, ys = ps[:, :, 0], ps[:, :, 1]
	min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)
	min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)
	return pr.create(min_x, min_y, max_x-min_x, max_y-min_y)

def _C(ps): 
	ndim = np.ndim(ps)
	if ndim == 1: assert len(ps) % 2 == 0
	elif ndim == 2: assert np.shape(ps)[1] % 2 == 0
	else: raise Exception("invalid polygon structure, dims is {}".format(ndim))
	return ps if isinstance(ps, np.ndarray) else np.asarray(ps)

def _B(ps): 
	ps = _C(ps)
	return ps[np.newaxis, :] if ps.ndim == 1 else ps
