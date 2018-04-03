from pyplus import codegen as cg

_TEMPLATE_FUNC = cg.F([
    cg.pydef('{F_PR}npoint', ['ps'], ['return {P}.shape[1] >> 1']),
    '',
    cg.pydef('{F_PR}bounding_rect', ['ps'], [
        'ps = {P}',
        'ps = ps.reshape(len(ps), -1 ,2)',
        'xs, ys = ps[:, :, 0], ps[:, :, 1]',
        'min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)',
        'min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)',
        'return rt.create(min_x, min_y, max_x-min_x, max_y-min_y)'
    ]),
    '',
    cg.pydef('{F_PR}bounding_pixel_rect', ['ps'], [
        'ps = {P}',
        'ps = ps.reshape(len(ps), -1 ,2)',
        'xs, ys = ps[:, :, 0], ps[:, :, 1]',
        'min_x, max_x = np.min(xs, axis=1), np.max(xs, axis=1)',
        'min_y, max_y = np.min(ys, axis=1), np.max(ys, axis=1)',
        'return pr.create(min_x, min_y, max_x-min_x, max_y-min_y)'
    ]),
    '',
    cg.pydef('{F_PR}IoU_pixel', ['ps1', 'ps2'], [
        'ps1 = {BP1}.astype(np.int64)',
        'ps2 = {BP2}.astype(np.int64)',
        'assert len(ps1) == len(ps2)',
        'rs = _bounding_pixel_rect(np.hstack([ps1, ps2])).reshape(len(ps1), -1)',
        'brs = pr.bottom_right(rs).reshape(len(ps1), -1)',
        'ious = []',
        cg.B('for i in range(len(rs)):', [
            'br = brs[i]',
            'm1, m2 = np.zeros((br[1]+1, br[0]+1)), np.zeros((br[1]+1, br[0]+1))',
            'cv2.fillPoly(m1, [ps1[i].reshape(-1, 2)], 1)',
            'cv2.fillPoly(m2, [ps2[i].reshape(-1, 2)], 1)',
            'm1, m2 = m1.astype(np.int64), m2.astype(np.int64)',
            'intersect = (m1 & m2).sum()',
            'union = (m1 | m2).sum()',
            'ious.append(0 if union == 0 else intersect / union)'
        ]),
        'return np.asarray(ious).squeeze()'
    ]),
])



SRC = cg.generate([
    cg.import_as('numpy', 'np'),
    cg.pyimport('cv2'),
    cg.from_import_as('pygeom', 'rectangle', 'rt'),
    cg.from_import_as('pygeom', 'pixel_rect', 'pr'),
    '',
    cg.pydef('create', ['ps', 'dtype=None'], ['return _C(ps).astype(dtype)']),
    '',
    cg.pydef('aspoly', ['ps'], ['return _C(ps)']),
    '',
    _TEMPLATE_FUNC.format(F_PR='', P='_B(ps)', BP1='_B(ps1)', BP2='_B(ps2)'),
    '',
    _TEMPLATE_FUNC.format(F_PR='_', P='ps', BP1='ps1', BP2='ps2'),
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
