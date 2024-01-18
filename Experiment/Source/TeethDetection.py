import csv
import glob
import os
import random
from time import time

import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from progress.bar import Bar
from scipy import ndimage as ndi
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

PATH = os.path.abspath(
    os.path.join(
        os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - copia"
    )
)
print(PATH)


def show(image):
    """
    Display an image using matplotlib.

    Args:
        image: The image to be displayed.

    Returns:
        None
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation="nearest")
    plt.show()


def readAllImagesPath(PATH):
    """
    Generates a list of all the paths to the images located in the specified directory.

    Parameters:
    - PATH (str): The path to the directory containing the images.

    Returns:
    - file_list (list): A list of strings representing the paths to the images.

    """
    file_list = []
    cantidad = len(glob.glob(PATH + "*"))
    procesing = Bar("Archive:", max=cantidad)
    bar = tqdm(os.walk(PATH))
    for dirName, subdirList, fileList in bar:
        for i in fileList:
            if ("JPG" in i) or ("jpg" in i):
                file_list.append(i)
            bar.set_description("Archive %s" % i)
        procesing.next()
    procesing.finish()
    return file_list


def resize_All_Images(file_list, PATHSRC, PATHRESIZE):
    """
    Resizes all images in the given file list and saves them to the specified destination directory.

    Args:
        file_list (list): A list of image file names.
        PATHSRC (str): The source directory path where the images are located.
        PATHRESIZE (str): The destination directory path where the resized images will be saved.

    Returns:
        None

    Raises:
        None
    """
    start_time = time()
    if not os.path.exists(PATHRESIZE):
        os.mkdir(PATHRESIZE)
    resizing = tqdm(file_list)
    for i in resizing:
        resizing.set_description("Re-scaling image %s" % i)
        src = PATHSRC + i
        image = cv.imread(src, 1)
        image = cv.resize(image, (600, 200))
        if not os.path.exists(PATHRESIZE + i):
            cv.imwrite(os.path.join(PATHRESIZE, i), image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Image re-scaling time: %0.10f seconds." % elapsed_time)


def readImages(PATH, fileList):
    """
    Reads a list of images from the specified directory.

    Parameters:
    - PATH (str): The path to the directory containing the images.
    - fileList (List[str]): A list of filenames of the images to be read.

    Returns:
    - images (List[numpy.ndarray]): A list of images read from the directory.

    This function reads each image specified in the fileList from the directory
    specified by the PATH parameter. It uses OpenCV's `cv.imread()` function to
    read each image and appends the resulting image to the `images` list. The
    function then returns the list of images.

    Note:
    - This function assumes that the images are stored in the specified
      directory in the correct format that can be read by OpenCV.
    - The function also prints the time taken to load the images to a vector.
    """
    start_time = time()
    images = []
    reading = tqdm(fileList)
    for i in reading:
        reading.set_description("Loading image %s" % i)
        src = PATH + i
        image = cv.imread(src)
        images.append(image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Time loading images to a vector: %0.10f seconds." % elapsed_time)
    return images


def imagesRGB2HSV(imagesRGB, directory, files):
    """
    Convert a list of RGB images to HSV format.

    Args:
        imagesRGB (list): A list of RGB images to be converted.
        directory (str): The directory where the converted images will be saved.
        files (list): A list of file names corresponding to the RGB images.

    Returns:
        list: A list of HSV images.
    """
    imagesHSV = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + "HSV/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = directory + "HSV/" + "RGB2HSV/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    HSV_ = Bar("Convert RGB to HSV:", max=len(files) * 4)
    for i, file in zip(imagesRGB, files):
        i = cv.cvtColor(i, cv.COLOR_RGB2HSV)
        imagesHSV.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + "dataset"
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + "/" + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        HSV_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0:
                colour[:, :, 0] = 0
            if l != 1:
                colour[:, :, 1] = 255
            if l != 2:
                colour[:, :, 2] = 255
            fil = "HSV_" + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            HSV_.next()
    HSV_.finish()
    return imagesHSV


def imagesBGR2HSV(images, directory, files):
    """
    Convert a list of BGR images to HSV format.

    Args:
        images (list): A list of BGR images to be converted.
        directory (str): The directory to save the converted images.
        files (list): A list of file names corresponding to the images.

    Returns:
        list: A list of images in HSV format.
    """
    images_ = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + "HSV/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = directory + "HSV/" + "BGR2HSV/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    HSV_ = Bar("Convert BGR to HSV:", max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2HSV)
        images_.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + "dataset"
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + "/" + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        HSV_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0:
                colour[:, :, 0] = 0
            if l != 1:
                colour[:, :, 1] = 255
            if l != 2:
                colour[:, :, 2] = 255
            fil = "HSV_" + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            HSV_.next()
    HSV_.finish()
    return images_


def imagesBGR2RGB(images, directory, files):
    """
    Converts a list of BGR images to RGB format.

    Args:
        images (List[numpy.ndarray]): A list of BGR images to be converted.
        directory (str): The directory path where the converted images will be saved.
        files (List[str]): A list of filenames corresponding to the images.

    Returns:
        List[numpy.ndarray]: A list of RGB images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    imagesR = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + "RGB/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    RGB_ = Bar("Convert BGR to RGB:", max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
        imagesR.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + "dataset"
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + "/" + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        RGB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0:
                colour[:, :, 0] = 0
            if l != 1:
                colour[:, :, 1] = 0
            if l != 2:
                colour[:, :, 2] = 0
            fil = "RGB_" + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            RGB_.next()
    RGB_.finish()
    return imagesR


def imagesBGR2YCR_CB(images, directory, files):
    """
    Convert a list of BGR images to YCR_CB color space.

    Args:
        images (List[numpy.ndarray]): The list of BGR images to be converted.
        directory (str): The directory where the converted images will be saved.
        files (List[str]): The list of filenames corresponding to the images.

    Returns:
        List[numpy.ndarray]: The list of YCR_CB images.

    Raises:
        None
    """
    imagesycr_cb = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + "YCR_CB/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    YCR_CB_ = Bar("Convert BGR to YCR_CB:", max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2YCR_CB)
        imagesycr_cb.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + "dataset"
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + "/" + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        YCR_CB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0:
                colour[:, :, 0] = 0
            if l != 1:
                colour[:, :, 1] = 0
            if l != 2:
                colour[:, :, 2] = 0
            fil = "YCR_CB_" + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            YCR_CB_.next()
    YCR_CB_.finish()
    return imagesycr_cb


def imagesRGB2YCR_CB(images, directory, files):
    """
    Converts a list of RGB images to YCrCb color space and saves the converted images to a specified directory.

    Args:
        images (list): A list of RGB images to be converted.
        directory (str): The directory where the converted images will be saved.
        files (list): A list of file names corresponding to the images.

    Returns:
        list: A list of YCrCb images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Example:
        imagesRGB2YCR_CB(images, directory, files)
    """
    imagesycr_cb = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + "RGB2YCR_CB/"
    if not os.path.exists(folder):
        os.mkdir(folder)
    YCR_CB_ = Bar("Convert BGR to YCR_CB:", max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_)
        imagesycr_cb.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + "dataset"
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + "/" + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        YCR_CB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0:
                colour[:, :, 0] = 0
            if l != 1:
                colour[:, :, 1] = 0
            if l != 2:
                colour[:, :, 2] = 0
            fil = "YCR_CB_" + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            YCR_CB_.next()
    YCR_CB_.finish()
    return imagesycr_cb


def imagesGaussianBlur(images_hsv, directory):
    """
    Apply Gaussian blur to a list of HSV images.

    Parameters:
        images_hsv (List[np.ndarray]): A list of numpy arrays representing HSV images.
        directory (str): The directory where the images are stored.

    Returns:
        List[np.ndarray]: The list of HSV images after applying Gaussian blur.
    """
    for i in images_hsv:
        i = cv.GaussianBlur(i, (5, 5), 0)
    return images_hsv


def extractFeatures(PATH, fileList):
    """
    Extracts features from a list of images and returns a matrix of image data.

    Args:
        PATH (str): The path of the directory containing the images.
        fileList (list): A list of image file names.

    Returns:
        list: A matrix of image data, where each row represents an image and each column represents a pixel value.
    """
    matrix_data = []
    bar = tqdm(fileList)
    for file in bar:
        bar.set_description("Processing Image %s" % file)
        src = PATH + file
        image = cv.imread(src, 1)
        image_f = []
        for row in image:
            for col in row:
                for pixel in col:
                    image_f.append(pixel)
        matrix_data.append(image_f)
    return matrix_data


def standar_matrix(matrix_data):
    """
    Generates a standardized matrix by padding each row with zeros to match the length of the longest row in the input matrix.

    Parameters:
    - matrix_data (list of lists): A matrix represented as a list of lists, where each inner list represents a row in the matrix.

    Returns:
    - maxi (list): A list containing the length of each row in the input matrix.

    Example:
    matrix_data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    standar_matrix(matrix_data) returns [3, 2, 4]
    """
    maxi = []
    for c in matrix_data:
        maxi.append(len(c))
    maximo = max(maxi)
    for i in matrix_data:
        if len(i) < maximo:
            dif = maximo - len(i)
            for j in range(dif):
                i.append(0)
    return maxi


def readlabels(PATH):
    """
    Read labels from a file.

    Parameters:
        PATH (str): The path to the directory containing the file.

    Returns:
        list: A list of strings representing the lines read from the file.
    """
    with open(os.path.join(PATH, "labels.txt")) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def show_hsv_hist(image):
    """
    Generate a histogram of the HSV color space for an input image.

    Parameters:
    - image: numpy.ndarray
        The input image to generate the histogram for.

    Returns:
    None
    """
    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i / 180, 1, 0.8)) for i in range(0, 180)]
    plt.bar(range(0, 180), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title("Hue")
    plt.show()
    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])
    colours = [colors.hsv_to_rgb((0, i / 256, 1)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title("Saturation")
    plt.show()

    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i / 256)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title("Value")
    plt.show()


def ImageSegmentation(images, directory, files):
    """
    Perform image segmentation on a list of images using a specified directory and file names.

    Parameters:
    - images: a list of images to be segmented
    - directory: a string representing the directory where the segmented images will be saved
    - files: a list of file names corresponding to the images

    Returns:
    None
    """
    start_time = time()
    src = directory + "SegmentationHSV"
    if not os.path.exists(src):
        os.mkdir(src)
    segmentation = Bar("Segmenting images:", max=len(images))
    for image, file in zip(images, files):
        ycrcbmin = np.array((170, 40, 235))
        ycrcbmax = np.array((180, 140, 245))
        ###imagebar
        show_hsv_hist(image)
        ###endbar
        skin_ycrcb = cv.inRange(image, ycrcbmin, ycrcbmax)
        kernel = np.ones((5, 5), np.uint8)
        img_erode = cv.erode(skin_ycrcb, kernel, iterations=1)
        holesimg = ndi.binary_fill_holes(img_erode).astype(np.int)
        imageio.imwrite(os.path.join(src, file), holesimg)
        segmentation.next()
    # break
    segmentation.finish()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Segmentation Time: %0.10f seconds." % elapsed_time)


def writefeats(PATH, matrix):
    """
    Write features to a CSV file.

    Args:
        PATH (str): The path to the directory where the CSV file will be created.
        matrix (list): A 2D list representing the matrix of features to be written to the CSV file.

    Returns:
        None
    """
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    src = PATH + "feats2.csv"
    headers = []
    for i in range(len(matrix[0])):
        headers.append("pixel" + str(i))
    with open(src, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(headers)
        writer.writerows(matrix)


def tooth_recognition(file, labels, images, filename):
    """
    Generates the function comment for the given function body in a markdown code block with the correct language syntax.

    Args:
        file (str): The path of the file.
        labels (list): The list of labels.
        images (list): The list of images.
        filename (list): The list of filenames.

    Returns:
        None
    """
    print("Classification")
    data = pd.read_csv(file + "feats2.csv").as_matrix()
    clf = DecisionTreeClassifier()
    xtrain = data[0:20, 0:]
    train_label = labels[:20]
    clf.fit(xtrain, train_label)
    xtest = data[20:, 0:]
    actual_label = labels[20]
    file_Random = random.randint(0, 20)
    d = xtest[file_Random]
    color = str(clf.predict([xtest[file_Random]]))
    print(clf.predict([xtest[file_Random]]))
    plt.imshow(images[file_Random])
    plt.title(filename[file_Random] + " color: " + color)
    plt.axis("off")
    plt.show()
    print("size xtest: ", len(d))

    p = clf.predict(xtest)
    count = 0
    for i in range(0, 20):
        count += 1 if p[i] == actual_label else 0
    print("Accuracy=", (count / 20) * 100)


def main():
    srcdataset = PATH + "DATASET - copia/"
    datasetresize = PATH + "ResizeDATASET/"
    directory_segmentation = PATH + "Segmentation/"
    directory_feats = PATH + "Features/"
    labels = readlabels(PATH)
    files = readAllImagesPath(PATH)
    # resizeAllImages(files, srcdataset, datasetresize)
    # images = readImages(datasetresize, files)
    # imagesRgb = imagesBGR2RGB(images, directory_segmentation, files)
    # imagesHsv=imagesBGR2HSV(images,directory_segmentation,files)
    # matrix_features = extractFeatures(datasetresize, files)
    # writefeats(directory_feats, matrix_features)
    # tooth_recognition(directory_feats, labels, imagesRgb,files)


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Program execution time: %0.10f seconds." % elapsed_time)
    os.system("PAUSE")
