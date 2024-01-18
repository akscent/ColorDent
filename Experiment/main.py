import errno
import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm

# sys.path.append("../Source")

import Source.Classification as Cl
import Source.FeatureExtraction as fE
import Source.PreProcessingData as pD
import Source.ReadImages as rI


def show(image):
    """
    Display an image in a new window and wait for a key press to close the window.

    Parameters:
        image (numpy.ndarray): The image to be displayed.

    Returns:
        None
    """
    cv.imshow("Imagen ", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


class MainClass:
    PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
    PATH_IMAGES_ORIGINAL = os.path.abspath(
        os.path.join(
            os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir),
            "DATASET - Original",
        )
    )
    PATH_Labels = os.path.abspath(
        os.path.join(
            os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"
        )
    )
    PATH_LabelsXML = os.path.abspath(
        os.path.join(
            os.path.join(
                os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"
            ),
            "LabelsXML",
        )
    )
    PATH_IMAGES = os.path.abspath(
        os.path.join(os.path.join(os.getcwd(), os.pardir), "DATASET")
    )
    readimages = None
    preprocessing = None
    featureExtraction = None
    clasification = None

    def __init__(self):
        """
        Initializes the object and creates the necessary directories for preprocessing, feature extraction, sampling, and classification.
        """
        try:
            if not os.path.exists(self.PROJECT_PATH + "PreProcessing"):
                os.mkdir(os.path.join(self.PROJECT_PATH, "PreProcessing"))
                print("Directory Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("PreProcessing Directory Already Exists.")
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + "FeatureExtraction"):
                os.mkdir(os.path.join(self.PROJECT_PATH, "FeatureExtraction"))
                print("Directory Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("FeatureExtraction Directory Already Exists.")
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + "Sampling"):
                os.mkdir(os.path.join(self.PROJECT_PATH, "Sampling"))
                print("Directory Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("Sampling Directory Already Exists.")
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + "Classification"):
                os.mkdir(os.path.join(self.PROJECT_PATH, "Classification"))
                print("Directory Classification Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("Classification Directory Already Exists.")
            else:
                raise
        self.readimages = rI.LoadData(self.PATH_IMAGES_ORIGINAL)
        self.preProcessing = pD.PreProcessingData(
            self.PROJECT_PATH, self.PATH_IMAGES_ORIGINAL
        )
        self.featureExtraction = fE.FeatureExtraction(
            self.PROJECT_PATH, self.PATH_IMAGES_ORIGINAL
        )
        self.clasification = Cl.Classification(self.PROJECT_PATH)

    def main_run(self):
        """
        This function is the entry point for running the main logic of the program.
        It performs various image processing operations on a given image.

        Parameters:
            self (object): An instance of the current class.

        Returns:
            None
        """
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction
        img, name = read.read_One_Image(self.PATH_IMAGES)
        height_ori, width_ori, depth_ori = img.shape
        img_resize = pp.resize_Image(img, name)
        img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
        hsv_image = pp.rgb_2_HSV(img_resize, name)
        stack, name = pp.stackColors(hsv_image, name)
        pp.hsv_hist(hsv_image, name)
        #
        # plt.imshow(img_resize)
        name, blurimage = pp.blurImage(img_resize, name)
        pp.show_mask(blurimage, name)
        pp.overlay_mask(blurimage, img_resize, name)
        # pp.getChromatiColor(img_resize,name,fe)
        # img_rgb2ycbcr = pp.rgb_2_YCrCb(img_resize, name)
        img_rgb2hsv = pp.rgb_2_HSV(img_resize, name)

    """easygui.msgbox("Image original shape: \n Height:" + str(height_ori) + "px, Width:" + str(width_ori) + "px" +
                   "\n Image Resize shape: \n Height:" + str(height_res) + "px, Width:" + str(width_res) + "px",
                   image=os.path.join(os.path.join(os.getcwd(), os.path.pardir),
                                      'PreProcessing/ResizeImages/' + name),
                   title="Image Shape - PreProcessing ")"""

    def savebin(self):
        """
        Save the preprocessed binarized images.
        """
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction
        images, names = read.read_Images(self.PATH_IMAGES_P)
        for image_point, name_point in zip(images, names):
            img_resize = pp.resize_Image(image_point, name_point)
            img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
            img_resize = cv.cvtColor(img_resize, cv.COLOR_RGB2GRAY)
            pp.bin(img_resize, name_point)

    def main_alldataset(self):
        """
        Executes the main functionality of the program for all datasets.

        This function allows the user to choose between two options:
        1. Read images and extract features.
        2. Read feature file and train algorithm.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None

        Raises:
        - None
        """
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction
        cc = self.clasification
        doption = input(
            "What model? \n1. Read images and get characteristics"
            "\n2. Read feature file and train algorithm\n"
        )
        option = int(doption)
        print(option)

        if option == 1:
            images, names = read.read_Images(self.PATH_IMAGES)
            bar = tqdm(images, ncols=len(images), unit=" image")
            for image_point, name_point in zip(bar, names):
                bar.set_description("Processing image %s" % name_point)
                img_resize = pp.resize_Image(image_point, name_point)
                img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
                hsv_image = pp.rgb_2_HSV(img_resize, name_point)
                stack, name_point = pp.stackColors(hsv_image, name_point)

                pp.hsv_hist(hsv_image, name_point)

                name_point, blurimage = pp.blurImage(img_resize, name_point)
                """file_ = open(os.path.join(self.PROJECT_PATH, 'Pruebas') + name_point + '.txt', "w")
                for i in blurimage:
                    file_.write(str(i))
                file_.close()"""

                mask = pp.findBiggestContour(blurimage, name_point)
                channelR, channelG, channelB = cv.split(img_resize)
                red = fe.getFeaturesVector(channelR, mask)
                green = fe.getFeaturesVector(channelG, mask)
                blue = fe.getFeaturesVector(channelB, mask)

                imagen = [red, green, blue]
                filefeaturespath = os.path.join(
                    os.path.join(self.PROJECT_PATH, "FeatureExtraction"), "features.csv"
                )
                fe.getFeatures(imagen, filefeaturespath, name_point)
                fileLabels = open(os.path.join(self.PATH_Labels, "Labels.csv"))
                pp.show_mask(blurimage, name_point)
                pp.overlay_mask(blurimage, img_resize, name_point)
        elif option == 2:
            times_execution = -1
            while times_execution <= 0:
                times_execution = int(
                    input("\nIndicate the number of execution times:")
                )
            bar = tqdm(range(times_execution), unit=" times")
            for i in bar:
                filefeaturespath = os.path.join(
                    os.path.join(self.PROJECT_PATH, "FeatureExtraction"), "features.csv"
                )
                names, features = cc.readfeatures(filefeaturespath)
                labels = cc.readLabels(self.PATH_Labels)
                features_images = features.values
                vals_to_replace = {"a1": "0", "a2": "1", "a3": "2", "a35": "3"}
                tags = ["0", "1", "2", "3"]
                target_names = ["a1", "a2", "a3", "a35"]
                folds = int(input("\nNumber of pages to separate the data set:"))
                test_size = int(input("\n Training/test data set split percentage:"))
                X, Y = cc.CrossValidation(features, labels, test_size / 100)
                SVM, DT, KNN = cc.classification(
                    self.PATH_IMAGES, X, Y, folds, tags, target_names, vals_to_replace
                )
                print("--------- Training ---------")
                for S, D, K in zip(SVM, DT, KNN):
                    (
                        matrix_confusion_SVM,
                        report_clasification_SVM,
                        report_scores_SVM,
                    ) = S.split()
                    (
                        matrix_confusion_DT,
                        report_clasification_DT,
                        report_scores_DT,
                    ) = D.split()
                    (
                        matrix_confusion_KNN,
                        report_clasification_KNN,
                        report_scores_KNN,
                    ) = K.split()
                    print("\n")
                    print("-------SVM------")
                    for report in report_clasification_SVM:
                        print(report)
                        for item in report:
                            print(report[item])
                    print("--------MEAN SVM--------")
                    print(np.mean(report_scores_SVM))
                    print("-------DT------")
                    for report in report_clasification_DT:
                        print(report)
                        for item in report:
                            print(report[item])
                    print("--------MEAN DT--------")
                    print(np.mean(report_scores_DT))

                    print("-------KNN------")
                    for report in report_clasification_KNN:
                        print(report)
                        for item in report:
                            print(report[item])
                    print("--------MEAN KNN--------")
                    print(np.mean(report_scores_KNN))
                    print("--------- TEST ---------")


if __name__ == "__main__":
    tesis: MainClass = MainClass()
    # tesis.main_run()
    tesis.main_alldataset()
    # tesis.savebin()
    print("The experiment has been completed")
    sys.exit(0)
