import os

import cv2 as cv
from tqdm import tqdm


class LoadData:
    PATH = None

    def __init__(self, PATH_IMAGES):
        self.PATH = PATH_IMAGES

    def read_Images(self, PATH):
        images = []
        names = []
        amount = len(os.listdir(PATH))
        bar = tqdm(os.listdir(PATH), ncols=amount, unit=' image')
        for i in bar:
            if ('JPG' in i) or ('jpg' in i):
                try:
                    img = cv.imread(os.path.join(PATH, i))
                    images.append(img)
                    names.append(i)
                except IOError:
                    print('Image cannot read corrctly: ' + i)
            bar.set_description("Leyendo archivo %s" % i)
        return images, names

    def read_One_Image(self, PATH):
        test_image = "101_0092.JPG"  # "100_0055.JPG"
        cristhian_mouth = "101_0000.JPG"
        name = test_image
        image_string = os.path.join(PATH, name)
        image = cv.imread(image_string, 1)
        return image, name