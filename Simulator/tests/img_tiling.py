#!/usr/bin/python3

from operator import le
import numpy as np
import cv2 as cv

def tile_image(image, x,y,w,h, rows, cols, tile_width, return_size=(32,32)):
    assert image.shape[0] >= y + h, f'Image height is {image.shape[0]} but y is {y} and h is {h}'
    assert image.shape[1] >= x + w, f'Image width is {image.shape[1]} but x is {x} and w is {w}'
    assert tile_width < w, f'Tile width is {tile_width} but w is {w}'
    #check x,y,w,h,rows,cols,tile_width are all ints
    assert isinstance(x, int), f'x is {x}'
    assert isinstance(y, int), f'y is {y}'
    assert isinstance(w, int), f'w is {w}'
    assert isinstance(h, int), f'h is {h}'
    assert isinstance(rows, int), f'rows is {rows}'
    assert isinstance(cols, int), f'cols is {cols}'
    assert isinstance(tile_width, int), f'tile_width is {tile_width}'
    img = image[y:y+h, x:x+w]
    region_width = int(1.0 * w / cols)
    region_height = int(1.0 * h / rows)
    centers_x = np.linspace(int(tile_width/2), w-int(tile_width/2), cols, dtype=int)
    centers_y = np.linspace(int(tile_width/2), h-int(tile_width/2), rows, dtype=int)
    # centers = np.stack([centers_x, centers_y], axis=1)
    imgs = [] 
    centers = []
    for i in range(rows):
        for j in range(cols):
            # img = image[y:y+h, x:x+w]
            im = img[centers_y[i]-tile_width//2:centers_y[i]+tile_width//2, centers_x[j]-tile_width//2:centers_x[j]+tile_width//2].copy()
            im = cv.resize(im, return_size)
            imgs.append(im)
            centers.append([x+centers_x[j], y+centers_y[i]])
    return imgs, centers 

if __name__ == '__main__':
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    img = cv.imread('img.png')

    ROWS = 8
    COLS = 8
    TILE_WIDTH = 64

    imgs, centers = tile_image(img, 320, 0, int(img.shape[1]/2), int(img.shape[0]/2), ROWS, COLS, TILE_WIDTH)
    print(centers)

    for i in range(len(imgs)):
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        img = cv.rectangle(img, (centers[i][0]-TILE_WIDTH//2, centers[i][1]-TILE_WIDTH//2), (centers[i][0]+TILE_WIDTH//2, centers[i][1]+TILE_WIDTH//2), color, 2)

    cv.imshow('image', img)
    cv.waitKey(1)
    print(img.shape)

    if len(centers) < 20:
        for i in range(len(imgs)):
            cv.namedWindow('img'+str(i), cv.WINDOW_NORMAL)
            cv.imshow('img' + str(i), imgs[i])
            # cv.waitKey(0)
    cv.waitKey(0)
    cv.destroyAllWindows()


