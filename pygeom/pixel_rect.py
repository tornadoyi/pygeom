import numpy as np
from pygeom import rectangle as rt
from pygeom.rectangle import * 

_DTYPES = [np.int, np.int8, np.int16, np.int32, np.int64, np.long, np.longlong]

def create(xs, ys, ws, hs, dtype=None): 
	assert np.shape(xs) == np.shape(ys) == np.shape(ws) == np.shape(hs)
	if dtype is not None: assert dtype in _DTYPES
	return np.hstack([
		np.asarray(xs, dtype=dtype).reshape(-1, 1),
		np.asarray(ys, dtype=dtype).reshape(-1, 1),
		np.asarray(ws, dtype=dtype).reshape(-1, 1),
		np.asarray(hs, dtype=dtype).reshape(-1, 1)]).squeeze()

def create_with_size(pts, sizes, dtype=None): 
	assert np.shape(pts) == np.shape(sizes)
	if dtype is not None: assert dtype in _DTYPES
	pts = np.asarray(pts, dtype=dtype)
	sizes = np.asarray(sizes, dtype=dtype)
	return np.hstack([
		np.asarray(pts, dtype=dtype).reshape(-1, 2),
		np.asarray(sizes, dtype=dtype).reshape(-1, 2)]).squeeze()


def bottom(rs): rs=_B(rs); return rs[:, 1] + rs[:, 3] - (rs[:, 3] > 0)

def right(rs): rs=_B(rs); return rs[:, 0] + rs[:, 2] - (rs[:, 2] > 0)

def center(rs): 
	rs = _B(rs)
	ws, hs = _width(rs), _height(rs)
	return pt.create(
		_left(rs) + (ws - (ws > 0)) >> 2,
		_top(rs) + (hs - (hs > 0)) >> 2,
		dtype=rs.dtype)


def _bottom(rs): return rs[:, 1] + rs[:, 3] - (rs[:, 3] > 0)

def _right(rs): return rs[:, 0] + rs[:, 2] - (rs[:, 2] > 0)

def _center(rs): 
	
	ws, hs = _width(rs), _height(rs)
	return pt.create(
		_left(rs) + (ws - (ws > 0)) >> 2,
		_top(rs) + (hs - (hs > 0)) >> 2,
		dtype=rs.dtype)



_B = rt._B
_width = rt._width
_height = rt._height
_left = rt._left
_top = rt._top
