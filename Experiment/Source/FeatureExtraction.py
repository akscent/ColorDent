import errno
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

import Source.PreProcessingData as pD
import Source.ReadImages as rI


class FeatureExtraction:
    readImages = None
    preProcessing = None

    def __init__(self, PATH_PROJECT, PATH_IMAGES):
        """
        Initializes the object with the given project and image paths.

        Parameters:
            PATH_PROJECT (str): The path of the project directory.
            PATH_IMAGES (str): The path of the images directory.

        Returns:
            None
        """
        self.path_project = os.path.join(PATH_PROJECT, "FeatureExtraction")
        self.path_dataset = PATH_IMAGES
        try:
            if not os.path.exists(self.path_project + "GetColors"):
                self.path_getColor = os.path.join(self.path_project, "GetColors")
                os.mkdir(self.path_getColor)

                print("ResizeImages Directory Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("ResizeImages Directory Already Exists.")
            else:
                raise
        self.preProcessing = pD.PreProcessingData(PATH_PROJECT, PATH_IMAGES)
        self.readImages = rI.LoadData(PATH_PROJECT)

    def RGB2HEX(self, color):
        """
        Convert an RGB color value to a hexadecimal color code.

        Parameters:
            color (list): A list containing the RGB values of the color. The RGB values should be integers in the range 0-255.

        Returns:
            str: The hexadecimal color code representing the input RGB color. The color code is in the format "#RRGGBB", where RR, GG, and BB represent the hexadecimal values of the red, green, and blue components of the color respectively.
        """
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def get_colors(self, src, number_of_colors, show_chart, name):
        """
        Get the dominant colors from an image.

        Parameters:
        - src: numpy array representing the image.
        - number_of_colors: int, the number of dominant colors to retrieve.
        - show_chart: bool, whether to display a pie chart of the dominant colors.
        - name: str, the name of the image.

        Returns:
        - rgb_colors: list, the dominant colors in RGB format.
        - hex_colors: list, the dominant colors in HEX format.
        """
        modified_image = src.reshape(src.shape[0] * src.shape[1], 3)
        clf = KMeans(n_clusters=number_of_colors)
        labels = clf.fit_predict(modified_image)
        counts = Counter(labels)
        center_colors = clf.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]

        if show_chart:
            path_image_name = os.path.join(self.path_getColor, name)
            plt.figure(figsize=(8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
            plt.savefig(path_image_name)
            plt.show()

        return rgb_colors, hex_colors

    def getFeaturesVector(self, image, mask):
        """
        Generates a feature vector based on the given image and mask.

        Parameters:
            image (ndarray): The input image.
            mask (ndarray): The mask to apply on the image.

        Returns:
            list: The feature vector generated from the image and mask.
        """
        features = []
        imagecpy = image.copy()
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if j == 255:
                    # print(image[i][j])
                    features.append(imagecpy[i][j])
        return features

    def meanVector(self, vector_caracteristicas):
        """
        Calculate the mean of a vector.

        Args:
            vector_caracteristicas (array-like): The vector of characteristics.

        Returns:
            float: The mean of the vector.
        """
        return np.mean(vector_caracteristicas)

    def varVector(self, vector_caracteristicas):
        """
        Calculate the moment of a given vector of characteristics.

        Args:
            vector_caracteristicas (list): A list of values representing the characteristics of the vector.

        Returns:
            float: The moment of the vector.

        """
        return stats.moment(vector_caracteristicas)

    def skewVector(self, vector_caracteristicas):
        """
        Calculate the skewness of a vector of characteristics.

        Parameters:
            vector_caracteristicas (list): A list of numerical values representing the characteristics.

        Returns:
            float: The skewness value of the vector.
        """
        return stats.skew(vector_caracteristicas)

    def getFeatures(self, imagen, filefeaturespath, name_point):
        """
        Generates a function comment for the given function body.

        Args:
            imagen (list): A list of images.
            filefeaturespath (str): The path to the file where the features will be saved.
            name_point (str): The name of the point.

        Returns:
            None
        """
        if not (os.path.exists(filefeaturespath) or os.path.isfile(filefeaturespath)):
            filefeatures = open(filefeaturespath, "w")
        else:
            filefeatures = open(filefeaturespath, "a")
        features = []
        for j in range(0, len(imagen)):
            features.append(self.meanVector(imagen[j]))
            features.append(self.varVector(imagen[j]))
            features.append(self.skewVector(imagen[j]))
        filefeatures.write(name_point)
        for item in range(len(features)):
            filefeatures.write(",%.6f" % features[item])
        filefeatures.write("\n")  # "," + label +'''
        filefeatures.close()
