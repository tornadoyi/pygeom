from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}x', ['p'], ['return {P}[:, 0]']),
    '',
    cg.pydef('{F_PR}y', ['p'], ['return {P}[:, 1]']),
])



SRC = cg.generate([
    cg.import_as('numpy', 'np'),
    '',
    cg.pydef('create', ['xs', 'ys', 'dtype=None'], [
        'assert np.shape(xs) == np.shape(ys)',
        'return np.hstack([np.asarray(xs, dtype=dtype).reshape(-1, 1),',
        'np.asarray(ys, dtype=dtype).reshape(-1, 1)]).squeeze()',
    ]),
    '',
    cg.pydef('aspoint', ['p'], ['return _C(p)']),
    '',
    _TEMPLATE_FUNC.format(F_PR='', P='_B(p)'),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', P='p'),
    '',
    cg.pydef('_C', ['p'], [
        'ndim = np.ndim(p)',
        'if ndim == 1: assert len(p) == 2',
        'elif ndim == 2: assert np.shape(p)[1] == 2',
        'else: raise Exception("invalid point structure, dims is {}".format(ndim))',
        'return p if isinstance(p, np.ndarray) else np.asarray(p)',
    ]),
    '',
    cg.pydef('_B', ['p'], ['return _C(p).reshape(-1, 2)']),

])
