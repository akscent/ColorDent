import errno
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt, colors
# from pygments.util import xrange


class PreProcessingData:
    path_resize = None
    path_RGB2YCrCb = None
    path_RGB2HSV = None
    path_RGB2Lab = None
    path_RGB2LAB = None
    path_segmentation = None
    path_BarPlot = None
    path_BarPlotCh1 = None
    path_BarPlotCh2 = None
    path_BarPlotCh3 = None
    path_Mask = None
    path_Mask_Overlay = None
    path_Inverse = None
    path_project = None
    path_dataset = None

    def __init__(self, PATH_PROJECT, PATH_DATASET):
    	"""
    	Initialize the object with the given project path and dataset path.
    	
    	Parameters:
    	- PATH_PROJECT (str): The path of the project.
    	- PATH_DATASET (str): The path of the dataset.
    	
    	Returns:
    	None
    	"""
        self.path_project = os.path.join(PATH_PROJECT, 'PreProcessing')
        self.path_dataset = PATH_DATASET
        try:
            if not os.path.exists(self.path_project + 'Segmentation'):
                self.path_segmentation = os.path.join(self.path_project, 'Segmentation')
                os.mkdir(self.path_segmentation)
                print('Segmentation Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Segmentation Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'BarPlot'):
                self.path_BarPlot = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'BarPlot')

                os.mkdir(self.path_BarPlot)
                print('BarPlot Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('BarPlot Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'BarPlotCh1'):
                self.path_BarPlotCh1 = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'BarPlotCh1')

                os.mkdir(self.path_BarPlotCh1)
                print('BarPlotCh1 Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('BarPlotCh1 Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'BarPlotCh2'):
                self.path_BarPlotCh2 = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'BarPlotCh2')

                os.mkdir(self.path_BarPlotCh2)
                print('BarPlotCh2 Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('BarPlotCh2 Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'BarPlotCh3'):
                self.path_BarPlotCh3 = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'BarPlotCh3')

                os.mkdir(self.path_BarPlotCh3)
                print('BarPlotCh3 Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('BarPlotCh3 Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'Mask'):
                self.path_Mask = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'Mask')

                os.mkdir(self.path_Mask)
                print('Mask Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('Mask Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'MaskOverlay'):
                self.path_Mask_Overlay = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'MaskOverlay')

                os.mkdir(self.path_Mask_Overlay)
                print('MaskOverlay Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('MaskOverlay Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(os.path.join(self.path_project, 'Segmentation') + 'Inverse'):
                self.path_Inverse = os.path.join(os.path.join(self.path_project, 'Segmentation'), 'Inverse')

                os.mkdir(self.path_Inverse)
                print('Inverse Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('Inverse Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'ResizeImages'):
                self.path_resize = os.path.join(self.path_project, 'ResizeImages')
                os.mkdir(self.path_resize)
                print('ResizeImages Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('ResizeImages Directory Already Exists.')
            else:
                raise

        try:
            if not os.path.exists(self.path_project + 'RGB2YCrCb'):
                self.path_RGB2YCrCb = os.path.join(self.path_project, 'RGB2YCrCb')
                os.mkdir(self.path_RGB2YCrCb)
                print('RGB2YCrCb Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('RGB2YCbCr Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'HSV'):
                self.path_RGB2HSV = os.path.join(self.path_project, 'HSV')
                os.mkdir(self.path_RGB2HSV)
                print('HSV Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('HSV Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'LAB'):
                self.path_RGB2LAB = os.path.join(self.path_project, 'LAB')
                os.mkdir(self.path_RGB2LAB)
                print('LAB Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('LAB Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'Lab_'):
                self.path_RGB2Lab = os.path.join(self.path_project, 'Lab_')
                os.mkdir(self.path_RGB2Lab)
                print('Lab_ Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:

                print('Lab_ Directory Already Exists.')
            else:
                raise

    def bin(self, img, name):
        """
        Convert the given image to binary using various thresholding techniques.
        
        Args:
            img (ndarray): The input image to be converted to binary.
            name (str): The name of the output file.
        
        Returns:
            None
        
        Raises:
            None
        """
        path_bin = os.path.join(self.path_Inverse, name)
        ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
        ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
        ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
        thresh6 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 37, 2)
        # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'ADAPTATIVE']
        # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh6]
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.savefig(path_bin)
        plt.close('all')

    def resize_Image(self, image, name):
        """
        Resizes an image using OpenCV and saves it to a specified path.

        Parameters:
            image (numpy.ndarray): The image to be resized.
            name (str): The name of the resized image.

        Returns:
            numpy.ndarray: The resized image.
        """
        img = cv.resize(image, dsize=None, fx=1 / 3, fy=1 / 3, interpolation=cv.INTER_LINEAR)
        # img = cv.resize(src=image,None, fx=1 / 3, fy=1 / 3)
        path_name_image = os.path.join(self.path_resize, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_YCrCb(self, image, name):
        """
        Converts an RGB image to YCrCb color space.

        Parameters:
            image (numpy.ndarray): The RGB image to be converted.
            name (str): The name of the image.

        Returns:
            numpy.ndarray: The YCrCb image.

        """
        img = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
        path_name_image = os.path.join(self.path_RGB2YCrCb, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_HSV(self, image, name):
        """
        Converts an RGB image to HSV format.

        Args:
            image (ndarray): The RGB image to convert.
            name (str): The name of the image.

        Returns:
            ndarray: The HSV image.

        """
        img = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
        path_name_image = os.path.join(self.path_RGB2HSV, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    '''def rgb_2_HSI(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_HS)
        path_name_image = os.path.join(self.path_RGB2HSI, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img'''

    def rgb_2_LAB(self, image, name):
        """
        Convert an RGB image to LAB color space.

        Parameters:
            image (numpy.ndarray): The input RGB image.
            name (str): The name of the image.

        Returns:
            numpy.ndarray: The image converted to LAB color space.
        """
        img = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        path_name_image = os.path.join(self.path_RGB2LAB, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_Lab(self, image, name):
        """
        Convert an RGB image to the Lab color space.

        Parameters:
            image (numpy.ndarray): The input RGB image.
            name (str): The name of the image.

        Returns:
            numpy.ndarray: The converted image in the Lab color space.
        """
        img = cv.cvtColor(image, cv.COLOR_RGB2Lab)
        path_name_image = os.path.join(self.path_RGB2Lab, name)
        print('Exitoso RGB2_LAB')
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def bgr_2_Lab(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2Lab)
        path_name_image = os.path.join(self.path_RGB2Lab, name)
        print('Exit RGB2_LAB')
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def getChromatiColor(self, image, name, fe):
        """
        Get the chromatic color of an image.

        Parameters:
            image (numpy.ndarray): The input image.
            name (str): The name of the image.
            fe (FeatureExtraction): An instance of the FeatureExtraction class.

        Returns:
            None
        """
        plt.savefig(
            os.path.join(os.path.join(os.getcwd(), os.path.pardir), 'FeatureExtraction/GetColors/Original' + name))
        plt.close('all')
        height_res, width_res, depth_res = image.shape
        print("name", name)
        print("Image Resize shape: \n Height:", height_res, ", Width:", width_res)
        rgbcolors, hexcolors = fe.get_colors(image, 15, True, 'plot_' + name)
        print("RGB:\n", rgbcolors, "\n", "hexcolors:\n", hexcolors)

    def stackColors(self, image, name):
        """
        Generate a stack of color images.

        Args:
            image (numpy.ndarray): The input image.
            name (str): The name of the image.

        Returns:
            tuple: A tuple containing the following elements:
                - numpy.ndarray: The stack of color images.
                - str: The name of the image.
        """
        namefoldersplit = str.split(name, '.')
        namefolder = namefoldersplit[0]
        images = []
        for i in [0, 1, 2]:
            colour = image.copy()
            if i != 0: colour[:, :, 0] = 0
            if i != 1: colour[:, :, 1] = 255
            if i != 2: colour[:, :, 2] = 255
            images.append(colour)
        hsv_stack = np.vstack(images)
        rgb_stack = cv.cvtColor(hsv_stack, cv.COLOR_HSV2RGB)
        plt.imshow(rgb_stack)
        plt.savefig(self.path_BarPlot + '\\BAR_StackColor_' + name)
        plt.close('all')
        return rgb_stack, name

    def hsv_hist(self, image, name):
        """
        Generate histograms for the hue, saturation, and value channels of an image.

        Parameters:
            image (array-like): The input image.
            name (str): The name of the image file.

        Returns:
            None
        """
        namefoldersplit = str.split(name, '.')
        namefolder = namefoldersplit[0]
        plt.figure(figsize=(20, 3))
        histr = cv.calcHist([image], [0], None, [180], [0, 180])
        plt.xlim([0, 180])
        colours = [colors.hsv_to_rgb((i / 180, 1, 0.9)) for i in range(0, 180)]
        colours = np.array(colours)
        histr = np.array(histr)
        # print(range(0,180))
        x = list(range(0, 180))
        # print(histr.shape)
        # print(len(x))
        # print(colours.shape)
        for i in x:
            plt.bar(x[i], histr[i], color=colours[i], edgecolor=colours[i], width=1)
            plt.title('Hue')
        plt.savefig(self.path_BarPlotCh1 + '\\BAR_CHART_HUE_' + name)

        # Saturation
        plt.figure(figsize=(20, 3))
        histr = cv.calcHist([image], [1], None, [256], [0, 256])
        plt.xlim([0, 256])
        x = list(range(0, 256))
        colours = [colors.hsv_to_rgb((0, i / 256, 1)) for i in range(0, 256)]

        for i in x:
            plt.bar(x[i], histr[i], color=colours[i], edgecolor=colours[i], width=1)
            # plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
            plt.title('Saturation')
        plt.savefig(self.path_BarPlotCh2 + '\\BAR_CHART_SATURATION_' + name)
        # Value
        plt.figure(figsize=(20, 3))
        histr = cv.calcHist([image], [2], None, [256], [0, 256])
        plt.xlim([0, 256])
        x = list(range(0, 256))
        colours = [colors.hsv_to_rgb((0, 1, i / 256)) for i in range(0, 256)]
        for i in (x):
            plt.bar(x[i], histr[i], color=colours[i], edgecolor=colours[i], width=1)
            # plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
            plt.title('Value')
        plt.savefig(self.path_BarPlotCh3 + '\\BAR_CHART_VALUE_' + name)
        plt.close('all')

    def show_mask(self, mask, name):
        """
        Display the given mask image and save it with the given name.

        Parameters:
            mask (numpy.ndarray): The mask image to be displayed.
            name (str): The name of the mask image.

        Returns:
            None
        """
        namefoldersplit = str.split(name, '.')
        namefolder = namefoldersplit[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray')
        plt.savefig(self.path_Mask + '\\MASK_' + name)
        plt.close('all')

    def overlay_mask(self, mask, image, name):
        """
        Overlay a mask onto an image and save the result.

        Parameters:
            mask (numpy.ndarray): The mask to overlay on the image.
            image (numpy.ndarray): The image onto which the mask will be overlaid.
            name (str): The name of the image file.

        Returns:
            None
        """
        namefoldersplit = str.split(name, '.')
        namefolder = namefoldersplit[0]
        rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        img = cv.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
        plt.imshow(img)
        plt.savefig(self.path_Mask_Overlay + '\\MASKOVERLAY_' + name)
        plt.close('all')

    def blurImage(self, image, name):
        """
        Applies a blurring effect to the input image and detects red objects.

        Parameters:
            image (numpy.ndarray): The input image.
            name (str): The name of the image.

        Returns:
            tuple: A tuple containing the name of the image and the resulting binary image with red objects detected.
        """
        image_blur = cv.GaussianBlur(image, (7, 7), 0)
        image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV)
        min_red = np.array([10, 0, 0])
        max_red = np.array([40, 255, 255])
        image_red1 = cv.inRange(image_blur_hsv, min_red, max_red)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        image_red_closed = cv.morphologyEx(image_red1, cv.MORPH_CLOSE, kernel)
        # Remove specks
        image_red_closed_then_opened = cv.morphologyEx(image_red_closed, cv.MORPH_OPEN, kernel)
        return name, image_red_closed_then_opened

    def findBiggestContour(self, image, name):
        """
        Find the biggest contour in the given image and return a mask with only the largest contour filled.

        Parameters:
            image (numpy.ndarray): The input image.
            name (str): The name of the image.

        Returns:
            numpy.ndarray: The mask with only the largest contour filled.
        """
        # Copy to prevent modification
        image = image.copy()
        contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # Isolate largest contour
        # print(name + ":" + str(max(contours, key=cv.contourArea)))
        biggest_contour = max(contours, key=cv.contourArea)
        # Draw just largest contour
        mask = np.zeros(image.shape, np.uint8)
        cv.drawContours(mask, [biggest_contour], -1, 255, -1)
        return mask
