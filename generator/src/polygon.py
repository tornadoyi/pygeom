from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}npoint', ['ps'], ['return {P}.shape[1] >> 1']),
    '',
    cg.pydef('{F_PR}bounding_rect', ['ps'], [
        'ps = {P}.reshape(len(ps), -1 ,2)',
        'xs, ys = ps[:, :, 0], ps[:, :, 1]',
        'min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)',
        'min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)',
        'return rt.create(min_x, min_y, max_x-min_x, max_y-min_y)'
    ]),
    '',
    cg.pydef('{F_PR}bounding_pixel_rect', ['ps'], [
        'ps = {P}.reshape(len(ps), -1 ,2)',
        'xs, ys = ps[:, :, 0], ps[:, :, 1]',
        'min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)',
        'min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)',
        'return pr.create(min_x, min_y, max_x-min_x, max_y-min_y)'
    ]),
])



SRC = cg.generate([
    cg.import_as('numpy', 'np'),
    cg.from_import_as('pygeom', 'rectangle', 'rt'),
    cg.from_import_as('pygeom', 'pixel_rect', 'pr'),
    '',
    cg.pydef('create', ['ps', 'dtype=None'], ['return _C(ps).astype(dtype)']),
    '',
    cg.pydef('aspoly', ['ps'], ['return _C(ps)']),
    '',
    _TEMPLATE_FUNC.format(F_PR='', P='_B(ps)'),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', P='ps'),
    '',
    cg.pydef('_C', ['ps'], [
        'ndim = np.ndim(ps)',
        'if ndim == 1: assert len(ps) % 2 == 0',
        'elif ndim == 2: assert np.shape(ps)[1] % 2 == 0',
        'else: raise Exception("invalid polygon structure, dims is {}".format(ndim))',
        'return ps if isinstance(ps, np.ndarray) else np.asarray(ps)'
    ]),
    '',
    cg.pydef('_B', ['ps'], [
        'ps = _C(ps)',
        'return ps[np.newaxis, :] if ps.ndim == 1 else ps'
    ]),

])
