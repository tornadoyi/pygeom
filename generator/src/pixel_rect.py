from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}bottom', ['rs'], ['{rs_Brs}return rs[:, 1] + rs[:, 3] - (rs[:, 3] > 0)']),
    '',
    cg.pydef('{F_PR}right', ['rs'], ['{rs_Brs}return rs[:, 0] + rs[:, 2] - (rs[:, 2] > 0)']),
    '',
    cg.pydef('{F_PR}center', ['rs'], [
        '{rs_Brs}',
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
    cg.from_import_as('pygeom', 'rectangle', 'rt'),
    cg.from_import('pygeom.rectangle', '*'),
    '',
    '_DTYPES = [np.int, np.int8, np.int16, np.int32, np.int64, np.long, np.longlong]',
    '',
    cg.pydef('create', ['xs', 'ys', 'ws', 'hs', 'dtype=None', 'keepdims=False'], [
        'assert np.shape(xs) == np.shape(ys) == np.shape(ws) == np.shape(hs)',
        'if dtype is not None: assert dtype in _DTYPES',
        cg.B('rs = np.hstack([', [
            'np.asarray(xs, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ys, dtype=dtype).reshape(-1, 1),',
            'np.asarray(ws, dtype=dtype).reshape(-1, 1),',
            'np.asarray(hs, dtype=dtype).reshape(-1, 1)])',
        ]),
        'return _D(rs, np.ndim(xs)) if keepdims  else rs.squeeze()',
    ]),
    '',
    cg.pydef('create_with_size', ['pts', 'sizes', 'dtype=None', 'keepdims=False'], [
        'assert np.shape(pts) == np.shape(sizes)',
        'if dtype is not None: assert dtype in _DTYPES',
        'pts = np.asarray(pts, dtype=dtype)',
        'sizes = np.asarray(sizes, dtype=dtype)',
        cg.B('rs = np.hstack([', [
            'np.asarray(pts, dtype=dtype).reshape(-1, 2),',
            'np.asarray(sizes, dtype=dtype).reshape(-1, 2)])',
        ]),
        'return _D(rs, np.ndim(pts)) if keepdims  else rs.squeeze()',
    ]),
    '',
    '',
    _TEMPLATE_FUNC.format(F_PR='', rs_Brs='rs = _B(rs); ', DFT_KEEPDIM='False'),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', rs_Brs='[DEL]', DFT_KEEPDIM='True').clear('[DEL]'),
    '',
    '',
    '_B = rt._B',
    '_D = rt._D',
    '_top = rt._top',
    '_left = rt._left',
    #'_bottom = rt.bottom',
    #'_right = rt._right',
    '_height = rt._height',
    '_width = rt._width',
    '_area = rt._area',
    #'_center = rt._center',
    '_top_left = rt._top_left',
    '_top_right = rt._top_right',
    '_bottom_left = rt._bottom_left',
    '_bottom_right = rt._bottom_right',

])