import numpy as np

def create(xs, ys, dtype=None, keepdims=False): 
	assert np.shape(xs) == np.shape(ys)
	ps = np.hstack([
		np.asarray(xs, dtype=dtype).reshape(-1, 1),
		np.asarray(ys, dtype=dtype).reshape(-1, 1)])
	return _D(ps, np.ndim(xs)) if keepdims else ps.squeeze()

def aspoint(p): return _C(p)

def x(p): return _B(p)[:, 0]

def y(p): return _B(p)[:, 1]

def _x(p): return p[:, 0]

def _y(p): return p[:, 1]

def _C(p): 
	ndim = np.ndim(p)
	if ndim == 1: assert len(p) == 2
	elif ndim == 2: assert np.shape(p)[1] == 2
	else: raise Exception("invalid point structure, dims is {}".format(ndim))
	return p if isinstance(p, np.ndarray) else np.asarray(p)

def _B(p): return _C(p).reshape(-1, 2)

def _D(p, dims): 
	if p.ndim == dims: return p
	if dims == 1: return p.reshape(2)
	elif dims == 2: return p.reshape(-1, 2)
	else: raise Exception("invalid expected dims {}".format(dims))
