from pygeom import polygon as pl



def test():
    p1 = pl.aspoly([0, 0, 9, 9, 4, 4])
    p2 = pl.aspoly([0, 9, 9, 9, 4, 0])

    print(pl.IoU_pixel(p1, p2))




if __name__ == '__main__':
    test()