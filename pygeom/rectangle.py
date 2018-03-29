import numpy as np
from . import point2d as pt

def create(xs, ys, ws, hs, dtype=None): 
	assert np.shape(xs) == np.shape(ys) == np.shape(ws) == np.shape(hs)
	return np.hstack([
		np.asarray(xs, dtype=dtype).reshape(-1, 1),
		np.asarray(ys, dtype=dtype).reshape(-1, 1),
		np.asarray(ws, dtype=dtype).reshape(-1, 1),
		np.asarray(hs, dtype=dtype).reshape(-1, 1)]).squeeze()

def create_with_size(pts, sizes, dtype=None): 
	assert np.shape(pts) == np.shape(sizes)
	pts = np.asarray(pts, dtype=dtype)
	sizes = np.asarray(sizes, dtype=dtype)
	return np.hstack([
		np.asarray(pts, dtype=dtype).reshape(-1, 2),
		np.asarray(sizes, dtype=dtype).reshape(-1, 2)]).squeeze()

def asrect(rs): return _C(rs)


def top(rs): return _B(rs)[:, 1]

def left(rs): return _B(rs)[:, 0]

def bottom(rs): rs=_B(rs); return rs[:, 1] + rs[:, 3]

def right(rs): rs=_B(rs); return rs[:, 0] + rs[:, 2]

def height(rs): rs=_B(rs); return _B(rs)[:, 3]

def width(rs): rs=_B(rs); return _B(rs)[:, 2]

def area(rs): rs=_B(rs); return _width(rs) * _height(rs)

def center(rs): rs=_B(rs); return pt.create(_left(rs) + _width(rs) / 2, _top(rs) + _height(rs) / 2, dtype=rs.dtype)

def top_left(rs): rs=_B(rs); return pt.create(_left(rs), _top(rs))

def top_right(rs): rs=_B(rs); pt.create(_right(rs), _top(rs))

def bottom_left(rs): rs=_B(rs); return pt.create(_left(rs), _bottom(rs))

def bottom_right(rs): rs=_B(rs); return pt.create(_right(rs), _bottom(rs))

def size(rs): rs=_B(rs); return pt.create(_width(rs), _height(rs))

def empty(rs): rs=_B(rs); return (_width(rs) <= 0 | _height(rs) <= 0)

def intersect(rs1, rs2): 
	assert np.shape(rs1) == np.shape(rs2)
	rs1, rs2 = _B(rs1), _B(rs2)
	xs = np.max([_left(rs1), _left(rs2)], axis=0)
	ys = np.max([_top(rs1), _top(rs2)], axis=0)
	
	min_r = np.min([_right(rs1), _right(rs2)], axis=0)
	min_b = np.min([_bottom(rs1), _bottom(rs2)], axis=0)
	
	ws = np.clip(min_r - xs + 1, 0, None)
	hs = np.clip(min_b - ys + 1, 0, None)
	
	return create(xs, ys, ws, hs)

def union(rs1, rs2): 
	assert np.shape(rs1) == np.shape(rs2)
	rs1, rs2 = _B(rs1), _B(rs2)
	xs = np.min([_left(rs1), _left(rs2)], axis=0)
	ys = np.min([_top(rs1), _top(rs2)], axis=0)
	
	max_r = np.max([_right(rs1), _right(rs2)], axis=0)
	max_b = np.max([_bottom(rs1), _bottom(rs2)], axis=0)
	
	ws = np.clip(max_r - xs + 1, 0, None)
	hs = np.clip(max_b - ys + 1, 0, None)
	
	return create(xs, ys, ws, hs)

def IoU(rs1, rs2): rs1, rs2 = _B(rs1), _B(rs2); return _intersect(rs1, rs2) / _union(rs1, rs2)

def contains(rs, ps): 
	rs, ps = _B(rs), pt._B(ps)
	x, y = pt.x(ps)[np.newaxis, :], pt.y(ps)[np.newaxis, :]
	l, r, t, b = left(rs)[:, np.newaxis], right(rs)[:, np.newaxis], top(rs)[:, np.newaxis], bottom(rs)[:, np.newaxis]
	return (x >= l) & (x <= r) & (y >= t) & (y <= b)

def clip_top_bottom(rs, min, max=None): 
	clip_t, clip_b = np.clip(top(rs), min, max), np.clip(bottom(rs), min, max)
	return create(left(rs), clip_t, width(rs), clip_b-clip_t+1, rs.dtype)

def clip_left_right(rs, min, max=None): 
	clip_l, clip_r = np.clip(left(rs), min, max), np.clip(right(rs), min, max)
	return create(clip_l, top(rs), clip_r - clip_l + 1, height(rs), rs.dtype)

def _top(rs): return rs[:, 1]

def _left(rs): return rs[:, 0]

def _bottom(rs): return rs[:, 1] + rs[:, 3]

def _right(rs): return rs[:, 0] + rs[:, 2]

def _height(rs): return rs[:, 3]

def _width(rs): return rs[:, 2]

def _area(rs): return _width(rs) * _height(rs)

def _center(rs): return pt.create(_left(rs) + _width(rs) / 2, _top(rs) + _height(rs) / 2, dtype=rs.dtype)

def _top_left(rs): return pt.create(_left(rs), _top(rs))

def _top_right(rs): pt.create(_right(rs), _top(rs))

def _bottom_left(rs): return pt.create(_left(rs), _bottom(rs))

def _bottom_right(rs): return pt.create(_right(rs), _bottom(rs))

def _size(rs): return pt.create(_width(rs), _height(rs))

def _empty(rs): return (_width(rs) <= 0 | _height(rs) <= 0)

def _intersect(rs1, rs2): 
	assert np.shape(rs1) == np.shape(rs2)
	
	xs = np.max([_left(rs1), _left(rs2)], axis=0)
	ys = np.max([_top(rs1), _top(rs2)], axis=0)
	
	min_r = np.min([_right(rs1), _right(rs2)], axis=0)
	min_b = np.min([_bottom(rs1), _bottom(rs2)], axis=0)
	
	ws = np.clip(min_r - xs + 1, 0, None)
	hs = np.clip(min_b - ys + 1, 0, None)
	
	return create(xs, ys, ws, hs)

def _union(rs1, rs2): 
	assert np.shape(rs1) == np.shape(rs2)
	
	xs = np.min([_left(rs1), _left(rs2)], axis=0)
	ys = np.min([_top(rs1), _top(rs2)], axis=0)
	
	max_r = np.max([_right(rs1), _right(rs2)], axis=0)
	max_b = np.max([_bottom(rs1), _bottom(rs2)], axis=0)
	
	ws = np.clip(max_r - xs + 1, 0, None)
	hs = np.clip(max_b - ys + 1, 0, None)
	
	return create(xs, ys, ws, hs)

def _IoU(rs1, rs2): return _intersect(rs1, rs2) / _union(rs1, rs2)

def _contains(rs, ps): 
	
	x, y = pt._x(ps)[np.newaxis, :], pt._y(ps)[np.newaxis, :]
	l, r, t, b = _left(rs)[:, np.newaxis], _right(rs)[:, np.newaxis], _top(rs)[:, np.newaxis], _bottom(rs)[:, np.newaxis]
	return (x >= l) & (x <= r) & (y >= t) & (y <= b)

def _clip_top_bottom(rs, min, max=None): 
	clip_t, clip_b = np.clip(_top(rs), min, max), np.clip(_bottom(rs), min, max)
	return create(_left(rs), clip_t, _width(rs), clip_b-clip_t+1, rs.dtype)

def _clip_left_right(rs, min, max=None): 
	clip_l, clip_r = np.clip(_left(rs), min, max), np.clip(_right(rs), min, max)
	return create(clip_l, _top(rs), clip_r - clip_l + 1, _height(rs), rs.dtype)

def _C(rs): 
	ndim = np.ndim(rs)
	if ndim == 1: assert len(rs) == 4
	elif ndim == 2: assert np.shape(rs)[1] == 4
	else: raise Exception("invalid rectangle structure, dims is {}".format(ndim))
	return rs if isinstance(rs, np.ndarray) else np.asarray(rs)

def _B(rs): return _C(rs).reshape(-1, 2)
