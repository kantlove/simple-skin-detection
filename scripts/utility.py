import numpy as np


def is_skin(pixel):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]
    # is skin if this pixel is white
    return r > 250 and g > 250 and b > 250


def bern_k(lamda, x):
    part_1 = np.power(lamda, x)
    part_2 = np.power(1 - lamda, np.subtract(1, x))
    return np.multiply(part_1, part_2)


def normal_k(mean, variance, x):
    part_1 = np.divide(1.0, np.sqrt(np.multiply(2 * np.pi, variance)))
    part_2 = np.multiply(-0.5, np.divide(np.power(np.subtract(x, mean), 2), variance))
    return np.multiply(part_1, np.exp(part_2))


def set_pixel_black(image, row, col):
    image.itemset((row, col, 0), 0)
    image.itemset((row, col, 1), 0)
    image.itemset((row, col, 2), 0)


def set_pixel_white(image, row, col):
    image.itemset((row, col, 0), 255)
    image.itemset((row, col, 1), 255)
    image.itemset((row, col, 2), 255)