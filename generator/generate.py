import os
import shutil

def create_file(file_path, contents=None):
    if os.path.isfile(file_path): os.remove(file_path)
    if contents is None: contents = []
    elif isinstance(contents, (list, tuple)): pass
    elif isinstance(contents, str): contents = [contents]
    else: raise Exception('invalid contents type: {}'.format(type(contents)))

    with open(file_path, 'w') as f:
        for c in contents:
            f.write(c)
            f.write('\n')


def create_pygeom(src_path):
    # create root path
    if os.path.isdir(src_path): shutil.rmtree(src_path)
    os.mkdir(src_path)

    # create init file of pygeom
    create_file(os.path.join(src_path, '__init__.py'), [
        'from . import point2d',
        'from . import rectangle',
        'from . import pixel_rect',
    ])

    # create point2d
    from generator.src import point2d
    create_file(os.path.join(src_path, 'point2d.py'), point2d.SRC)

    # create rectangle
    from generator.src import rectangle
    create_file(os.path.join(src_path, 'rectangle.py'), rectangle.SRC)

    # create rectangle
    from generator.src import pixel_rect
    create_file(os.path.join(src_path, 'pixel_rect.py'), pixel_rect.SRC)




if __name__ == '__main__':
    # create pygeom path
    pygeom_path = os.path.join(os.path.dirname(__file__), '../pygeom')
    create_pygeom(pygeom_path)


