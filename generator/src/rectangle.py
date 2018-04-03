from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}top', ['rs'], ['return {_Brs}[:, 1]']),
    '',
    cg.pydef('{F_PR}left', ['rs'], ['return {_Brs}[:, 0]']),
    '',
    cg.pydef('{F_PR}bottom', ['rs'], ['{rs_Brs}return rs[:, 1] + rs[:, 3]']),
    '',
    cg.pydef('{F_PR}right', ['rs'], ['{rs_Brs}return rs[:, 0] + rs[:, 2]']),
    '',
    cg.pydef('{F_PR}height', ['rs'], ['{rs_Brs}return {_Brs}[:, 3]']),
    '',
    cg.pydef('{F_PR}width', ['rs'], ['{rs_Brs}return {_Brs}[:, 2]']),
    '',
    cg.pydef('{F_PR}area', ['rs'], ['{rs_Brs}return _width(rs) * _height(rs)']),
    '',
    cg.pydef('{F_PR}center', ['rs'], ['{rs_Brs}return pt.create(_left(rs) + _width(rs) / 2, _top(rs) + _height(rs) / 2, dtype=rs.dtype)']),
    '',
    cg.pydef('{F_PR}top_left', ['rs'], ['{rs_Brs}return pt.create(_left(rs), _top(rs))']),
    '',
    cg.pydef('{F_PR}top_right', ['rs'], ['{rs_Brs}return pt.create(_right(rs), _top(rs))']),
    '',
    cg.pydef('{F_PR}bottom_left', ['rs'], ['{rs_Brs}return pt.create(_left(rs), _bottom(rs))']),
    '',
    cg.pydef('{F_PR}bottom_right', ['rs'], ['{rs_Brs}return pt.create(_right(rs), _bottom(rs))']),
    '',
    cg.pydef('{F_PR}size', ['rs'], ['{rs_Brs}return pt.create(_width(rs), _height(rs))']),
    '',
    cg.pydef('{F_PR}empty', ['rs'], ['{rs_Brs}return (_width(rs) <= 0 | _height(rs) <= 0)']),
    '',
    cg.pydef('{F_PR}intersect', ['rs1', 'rs2', 'keepdim={DFT_KEEPDIM}'], [
        'assert np.shape(rs1) == np.shape(rs2)',
        'dims = np.ndim(rs1)',
        '{rs1_Brs1}',
        '{rs2_Brs2}',
        'xs = np.max([_left(rs1), _left(rs2)], axis=0)',
        'ys = np.max([_top(rs1), _top(rs2)], axis=0)',
        '',
        'min_r = np.min([_right(rs1), _right(rs2)], axis=0)',
        'min_b = np.min([_bottom(rs1), _bottom(rs2)], axis=0)',
        '',
        'ws = np.clip(min_r - xs + 1, 0, None)',
        'hs = np.clip(min_b - ys + 1, 0, None)',
        '',
        'rs = create(xs, ys, ws, hs)',
        'return _D(rs, dims) if keepdim else rs',
    ]),
    '',
    cg.pydef('{F_PR}union', ['rs1', 'rs2', 'keepdim={DFT_KEEPDIM}'], [
        'assert np.shape(rs1) == np.shape(rs2)',
        'dims = np.ndim(rs1)',
        '{rs1_Brs1}',
        '{rs2_Brs2}',
        'xs = np.min([_left(rs1), _left(rs2)], axis=0)',
        'ys = np.min([_top(rs1), _top(rs2)], axis=0)',
        '',
        'max_r = np.max([_right(rs1), _right(rs2)], axis=0)',
        'max_b = np.max([_bottom(rs1), _bottom(rs2)], axis=0)',
        '',
        'ws = np.clip(max_r - xs + 1, 0, None)',
        'hs = np.clip(max_b - ys + 1, 0, None)',
        '',
        'rs = create(xs, ys, ws, hs)',
        'return _D(rs, dims) if keepdim else rs',
    ]),
    '',
    cg.pydef('{F_PR}IoU', ['rs1, rs2', 'keepdim={DFT_KEEPDIM}'], [
        '{rs1_Brs1}',
        '{rs2_Brs2}',
        'return _intersect(rs1, rs2, keepdim=keepdim) / _union(rs1, rs2, keepdim=keepdim)']),
    '',
    cg.pydef('{F_PR}contains', ['rs, ps'], [
        '{rs_Brs}',
        '{ps_Bps}',
        'x, y = pt._x(ps)[np.newaxis, :], pt._y(ps)[np.newaxis, :]',
        'l, r, t, b = _left(rs)[:, np.newaxis], _right(rs)[:, np.newaxis], _top(rs)[:, np.newaxis], _bottom(rs)[:, np.newaxis]',
        'return (x >= l) & (x <= r) & (y >= t) & (y <= b)'
    ]),
    '',
    cg.pydef('{F_PR}clip_top_bottom', ['rs, min', 'max=None', 'keepdim={DFT_KEEPDIM}'], [
        'dims = np.ndim(rs)',
        '{rs_Brs}',
        'clip_t, clip_b = np.clip(_top(rs), min, max), np.clip(_bottom(rs), min, max)',
        'nrs = create(_left(rs), clip_t, _width(rs), clip_b-clip_t+1, rs.dtype)',
        'return _D(nrs, dims) if keepdim else nrs',
    ]),
    '',
    cg.pydef('{F_PR}clip_left_right', ['rs, min', 'max=None', 'keepdim={DFT_KEEPDIM}'], [
        'dims = np.ndim(rs)',
        '{rs_Brs}',
        'clip_l, clip_r = np.clip(_left(rs), min, max), np.clip(_right(rs), min, max)',
        'nrs = create(clip_l, _top(rs), clip_r - clip_l + 1, _height(rs), rs.dtype)',
        'return _D(nrs, dims) if keepdim else nrs',
    ]),

])



SRC = cg.generate([
    cg.import_as('numpy', 'np'),
    cg.from_import_as('pygeom', 'point2d', 'pt'),
    '',
    cg.pydef('create', ['xs', 'ys', 'ws', 'hs', 'dtype=None', 'keepdims=False'], [
        'assert np.shape(xs) == np.shape(ys) == np.shape(ws) == np.shape(hs)',
        cg.B('rs = np.hstack([', [
            'np.asarray(xs, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ys, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ws, dtype=dtype).reshape(-1, 1),',
            'np.asarray(hs, dtype=dtype).reshape(-1, 1)])'
        ]),
        'return _D(rs, np.ndim(xs)) if keepdims  else rs.squeeze()',
    ]),
    '',
    cg.pydef('create_with_size', ['pts', 'sizes', 'dtype=None', 'keepdims=False'], [
        'assert np.shape(pts) == np.shape(sizes)',
        'pts = np.asarray(pts, dtype=dtype)',
        'sizes = np.asarray(sizes, dtype=dtype)',
        cg.B('rs = np.hstack([', [
            'np.asarray(pts, dtype=dtype).reshape(-1, 2),',
            'np.asarray(sizes, dtype=dtype).reshape(-1, 2)])',
        ]),
        'return _D(rs, np.ndim(pts)) if keepdims  else rs.squeeze()',
    ]),
    '',
    cg.pydef('asrect', ['rs'], ['return _C(rs)']),
    '',
    '',
    _TEMPLATE_FUNC.format(F_PR='', _Brs='_B(rs)', rs_Brs='rs = _B(rs); ',
                          rs1_Brs1='rs1 = _B(rs1)', rs2_Brs2='rs2 = _B(rs2)',
                          ps_Bps='ps = pt._B(ps)', DFT_KEEPDIM='False'),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', _Brs='rs', rs_Brs='[DEL]',
                          rs1_Brs1='[DEL]', rs2_Brs2='[DEL]',
                          ps_Bps='[DEL]', DFT_KEEPDIM='True').clear('[DEL]'),
    '',
    cg.pydef('_C', ['rs'], [
        'ndim = np.ndim(rs)',
        'if ndim == 1: assert len(rs) == 4',
        'elif ndim == 2: assert np.shape(rs)[1] == 4',
        'else: raise Exception("invalid rectangle structure, dims is {}".format(ndim))',
        'return rs if isinstance(rs, np.ndarray) else np.asarray(rs)',
    ]),
    '',
    cg.pydef('_B', ['rs'], ['return _C(rs).reshape(-1, 4)']),
    '',
    cg.pydef('_D', ['rs', 'dims'], [
        'if rs.ndim == dims: return rs',
        'if dims == 1: return rs.reshape(4)',
        'elif dims == 2: return rs.reshape(-1, 4)',
        'else: raise Exception("invalid expected dims {}".format(dims))'
    ])

])