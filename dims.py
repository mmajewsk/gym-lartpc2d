import itertools
import math

import numpy as np

def neighborhood2d(size=3, ommit_center=False):
	"""
	Creates list of neighbourhood square indeces vectors. E.g:
    In[3]: neighborhood2d(3)
    Out[3]:
    array([[-1, -1],
           [-1,  0],
           [-1,  1],
           [ 0, -1],
           [ 0,  0],
           [ 0,  1],
           [ 1, -1],
           [ 1,  0],
           [ 1,  1]])
	:param size: size of the square
	:return:
	"""
	br = list(map(lambda x: x - math.floor(size / 2), range(size)))
	indeces = np.array(list(itertools.product(*[br, br])))
	if ommit_center:
		indeces = indeces[~np.all(indeces == 0, axis=1)]
	return indeces





def neighborhood3d(size=3):
	"""
	Creates list of neighbourhood cuve indeces vectors. E.g:
    In[3]: neighborhood3d(3)
    Out[3]:
    array([[-1, -1, -1],
           [-1, -1,  0],
           [-1, -1,  1],
           [-1,  0, -1],
           [-1,  0,  0],
           [-1,  0,  1],
           [-1,  1, -1],
           [-1,  1,  0],
           [-1,  1,  1],
           [ 0, -1, -1],
           [ 0, -1,  0],
           [ 0, -1,  1],
           [ 0,  0, -1],
           [ 0,  0,  0],
           [ 0,  0,  1],
           [ 0,  1, -1],
           [ 0,  1,  0],
           [ 0,  1,  1],
           [ 1, -1, -1],
           [ 1, -1,  0],
           [ 1, -1,  1],
           [ 1,  0, -1],
           [ 1,  0,  0],
           [ 1,  0,  1],
           [ 1,  1, -1],
           [ 1,  1,  0],
           [ 1,  1,  1]])
	:param size: size of the cube
	:return:
	"""
	br = list(map(lambda x: x - math.floor(size / 2), range(size)))
	return np.array(list(itertools.product(*[br, br, br])))

def get_range2d(arr, center_xy, low, high):
	assert center_xy.dtype.kind == 'i', \
		"The indeces are of type {}, expected int".format(center_xy.dtype)
	return arr[
			center_xy[0] - low : center_xy[0] + high,
			center_xy[1] - low : center_xy[1] + high,
			]

def set_range2d(arr, value, center_xy, low, high):
	assert center_xy.dtype.kind == 'i', \
		"The indeces are of type {}, expected int".format(center_xy.dtype)
	arr[
		center_xy[0] - low : center_xy[0] + high,
		center_xy[1] - low : center_xy[1] + high,
		] = value


def get_range3d(arr, center_xyz, low, high):
	assert center_xyz.dtype.kind == 'i', \
		"The indeces are of type {}, expected int".format(center_xyz.dtype)
	return arr[
			center_xyz[0] - low : center_xyz[0] + high,
			center_xyz[1] - low : center_xyz[1] + high,
			center_xyz[2] - low : center_xyz[2] + high,
			]

def set_range3d(arr, value, center_xyz, low, high):
	assert center_xyz.dtype.kind == 'i', \
		"The indeces are of type {}, expected int".format(center_xyz.dtype)
	arr[
		center_xyz[0] - low : center_xyz[0] + high,
		center_xyz[1] - low : center_xyz[1] + high,
		center_xyz[2] - low : center_xyz[2] + high,
		] = value


def cube3d(arr, center_xyz):
	return get_range3d(arr, center_xyz, 1, 2)

def add_frame(cube, frame=1):
    l = cube.shape[0]
    assert cube.shape[0]==cube.shape[1] and cube.shape[2]==cube.shape[0]
    s = cube.shape
    new_l = l+2*frame
    new_cube = np.zeros((new_l,)*3, dtype=cube.dtype)
    new_cube[frame:new_l-frame, frame:new_l-frame, frame:new_l-frame] = cube
    assert new_cube.shape[0] == l+2*frame
    return new_cube