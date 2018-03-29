from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}bottom', ['rs'], ['{BR_s}return rs[:, 1] + rs[:, 3] - (rs[:, 3] > 0)']),
    '',
    cg.pydef('{F_PR}right', ['rs'], ['{BR_s}return rs[:, 0] + rs[:, 2] - (rs[:, 2] > 0)']),
    '',
    cg.pydef('{F_PR}center', ['rs'], [
        '{BR}',
        'ws, hs = _width(rs), _height(rs)',
        cg.B('return pt.create(', [
            '_left(rs) + (ws - (ws > 0)) >> 2,',
            '_top(rs) + (hs - (hs > 0)) >> 2,',
            'dtype=rs.dtype)'
        ])
    ]),
    '',
])



SRC = cg.generate([
    cg.import_as('numpy', 'np'),
    cg.from_import_as('.', 'rectangle', 'rt'),
    cg.from_import('.rectangle', '*'),
    '',
    '_DTYPES = [np.int, np.int8, np.int16, np.int32, np.int64, np.long, np.longlong]',
    '',
    cg.pydef('create', ['xs', 'ys', 'ws', 'hs', 'dtype=None'], [
        'assert np.shape(xs) == np.shape(ys) == np.shape(ws) == np.shape(hs)',
        'if dtype is not None: assert dtype in _DTYPES',
        cg.B('return np.hstack([', [
            'np.asarray(xs, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ys, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ws, dtype=dtype).reshape(-1, 1),',
            'np.asarray(hs, dtype=dtype).reshape(-1, 1)]).squeeze()'
        ]),
    ]),
    '',
    cg.pydef('create_with_size', ['pts', 'sizes', 'dtype=None'], [
        'assert np.shape(pts) == np.shape(sizes)',
        'if dtype is not None: assert dtype in _DTYPES',
        'pts = np.asarray(pts, dtype=dtype)',
        'sizes = np.asarray(sizes, dtype=dtype)',
        cg.B('return np.hstack([', [
            'np.asarray(pts, dtype=dtype).reshape(-1, 2),',
            'np.asarray(sizes, dtype=dtype).reshape(-1, 2)]).squeeze()',
        ]),
    ]),
    '',
    '',
    _TEMPLATE_FUNC.format(F_PR='',
                          R='_B(rs)', BR='rs = _B(rs)', BR2='rs1, rs2 = _B(rs1), _B(rs2)', BRP='rs, ps = _B(rs), pt._B(ps)',
                          BR_s='rs=_B(rs); ', BR2_s='rs1, rs2 = _B(rs1), _B(rs2); '),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', CF_PR='_', R='rs', BR='', BR2='', BRP='', BR_s='', BR2_s=''),
    '',
    '',
    '_B = rt._B',
    '_width = rt._width',
    '_height = rt._height',
    '_left = rt._left',
    '_top = rt._top',

])