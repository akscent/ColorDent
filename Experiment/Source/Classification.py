import errno
import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Classification:
    def __init__(self, PATH_PROJECT):
        """
        Initializes the object with the given project path.

        Args:
            PATH_PROJECT (str): The path of the project.

        Returns:
            None
        """
        self.path_project = os.path.join(PATH_PROJECT, "Classification")
        try:
            if not os.path.exists(self.path_project + "SVM"):
                self.path_getColor = os.path.join(self.path_project, "SVM")
                os.mkdir(self.path_getColor)
                print("SVM Directory Created")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("SVM Directory Already Exists.")
            else:
                raise

    def readfeatures(self, features_Path):
        """
        Reads features from a specified file path.

        Args:
            features_Path (str): The path to the features file.

        Returns:
            tuple: A tuple containing the names and features read from the file.
        """
        # featuresFile = open(features_Path, "r")
        # featuresData = csv.reader(featuresFile)
        featuresFile = pd.read_csv(features_Path, sep=",", header=None)
        names = featuresFile.iloc[:, 0]
        features = featuresFile.iloc[:, 1:]
        shapefile = featuresFile.shape
        col = []
        for x in range(0, shapefile[1]):
            if x == 0:
                col.append("NAME")
            else:
                col.append("VALOR-" + str(x))
        featuresFile.columns = col
        # print(featuresFile)
        return names, features

    def readLabels(self, labels_path):
        """
        Reads the labels from a CSV file.

        Args:
            labels_path (str): The path to the directory containing the CSV file.

        Returns:
            pandas.DataFrame: The labels read from the CSV file.
        """
        labels_path = os.path.join(labels_path, "Labels.csv")
        labels = pd.read_csv(labels_path, sep=",", header=[0])
        return labels

    def classificatorSVM(self, features, labels):
        """
        Trains a Support Vector Machine classifier using the given features and labels.

        Parameters:
            features (list): A list of feature vectors.
            labels (list): A list of corresponding labels for each feature vector.

        Returns:
            svm.SVC: The trained Support Vector Machine classifier.
        """
        X = []
        for f in features:
            X.append(f)
        tags = []
        for tag in labels:
            tags.append(tag)
        clf = svm.SVC(gamma="scale")
        clf.fit(X, tags)
        return clf

    def DecisionTree(self, features, labels):
        """
        Builds a decision tree classifier using the provided features and labels.

        Parameters:
            features (list): A list of feature vectors.
            labels (list): A list of corresponding labels for each feature vector.

        Returns:
            DecisionTreeClassifier: A trained decision tree classifier.
        """
        dt = DecisionTreeClassifier(random_state=30, max_depth=300)
        X = []
        for f in features:
            X.append(f)
        tags = []
        for tag in labels:
            tags.append(tag)
        dt.fit(X, tags)

        return dt

    def KNN(self, features, labels):
        """
        Fits a K-Nearest Neighbors classifier to the given features and labels.

        Parameters:
            features (list): A list of input features.
            labels (list): A list of corresponding labels for the features.

        Returns:
            KNeighborsClassifier: The trained K-Nearest Neighbors classifier.
        """
        knn = KNeighborsClassifier(
            n_neighbors=200, algorithm="auto", weights="distance", n_jobs=-1
        )
        X = []
        for f in features:
            X.append(f)
        tags = []
        for tag in labels:
            tags.append(tag)
        knn.fit(X, tags)
        return knn

    def classification(
        self,
        path_dataset,
        features,
        labels,
        n_splits,
        tags,
        target_names,
        vals_to_replace,
    ):
        """
        This function performs a classification task using SVM, KNN, and Decision Tree classifiers.

        Parameters:
        - path_dataset: The path to the dataset directory.
        - features: A list of features.
        - labels: A list of labels.
        - n_splits: The number of splits for K-Fold cross-validation.
        - tags: A list of tags.
        - target_names: A list of target names.
        - vals_to_replace: A dictionary of values to replace in the 'Color' column.

        Returns:
        - SVM: A list containing the confusion matrix, classification report, and accuracy score for SVM classifier.
        - KNN: A list containing the confusion matrix, classification report, and accuracy score for KNN classifier.
        - DT: A list containing the confusion matrix, classification report, and accuracy score for Decision Tree classifier.
        """
        pd.options.mode.chained_assignment = None
        onlyfiles = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
        print(onlyfiles)
        k_folds = KFold(n_splits=n_splits)
        SVM = []
        KNN = []
        DT = []
        for stage, feature in zip(labels, features):
            stage["Color"] = stage["Color"].map(vals_to_replace)
            labels_color = stage["Color"].to_numpy().tolist()
            images_name = stage["Number of images"].to_numpy().tolist()
            svm_training_Score = []
            confusion_matrix_svm = []
            confusion_matrix_dt = []
            confusion_matrix_knn = []
            classification_report_svm = []
            classification_report_dt = []
            classification_report_knn = []
            score_accuracy_SVM = []
            score_accuracy_DT = []
            score_accuracy_KNN = []
            index_images_name = []
            k_folds.get_n_splits(index_images_name)
            print(list(images_name.index))

            for train_index, test_index in k_folds.split(images_name):
                train_label = []
                test_label = []
                train_features = []
                test_features = []
                for i in train_index:
                    train_features.append(
                        feature.to_numpy().tolist()[
                            images_name.index(onlyfiles[i].split(".")[0])
                        ]
                    )
                    # train_features.append(feature.to_numpy()[images_name.str(onlyfiles[i].split('.')[0])])
                    train_label.append(
                        labels_color[images_name.index(onlyfiles[i].split(".")[0])]
                    )
                SVM_Classifier = self.classificatorSVM(train_features, train_label)
                DT_Classifier = self.DecisionTree(train_features, train_label)
                KNN_Classifier = self.KNN(train_features, train_label)

                for i in test_index:
                    test_features.append(
                        feature.to_numpy()[
                            images_name.index(str(onlyfiles[i].split(".")[0]))
                        ]
                    )
                    test_label.append(
                        labels_color[images_name.index(str(onlyfiles[i].split(".")[0]))]
                    )
                predict_label_SVM = SVM_Classifier.predict(test_features)
                predict_label_DT = DT_Classifier.predict(test_features)
                predict_label_KNN = KNN_Classifier.predict(test_features)

                confusionMatrixSVM = confusion_matrix(test_label, predict_label_SVM)
                confusionMatrixDT = confusion_matrix(test_label, predict_label_DT)
                confusionMatrixKNN = confusion_matrix(test_label, predict_label_KNN)

                classification_report_svm.append(
                    classification_report(
                        test_label,
                        predict_label_SVM,
                        labels=tags,
                        target_names=target_names,
                        sample_weight=None,
                        digits=5,
                        output_dict=False,
                    )
                )
                classification_report_dt.append(
                    classification_report(
                        test_label,
                        predict_label_DT,
                        labels=tags,
                        target_names=target_names,
                        sample_weight=None,
                        digits=5,
                        output_dict=False,
                    )
                )
                classification_report_knn.append(
                    classification_report(
                        test_label,
                        predict_label_KNN,
                        labels=tags,
                        target_names=target_names,
                        sample_weight=None,
                        digits=5,
                        output_dict=False,
                    )
                )

                confusion_matrix_svm.append(confusionMatrixSVM)
                confusion_matrix_dt.append(confusionMatrixDT)
                confusion_matrix_knn.append(confusionMatrixKNN)

                score_accuracy_SVM.append(accuracy_score(test_label, predict_label_SVM))
                score_accuracy_DT.append(accuracy_score(test_label, predict_label_DT))
                score_accuracy_KNN.append(accuracy_score(test_label, predict_label_KNN))
            SVM_RESULTS = [
                confusion_matrix_svm,
                classification_report_svm,
                score_accuracy_SVM,
            ]
            DT_RESULTS = [
                confusion_matrix_dt,
                classification_report_dt,
                score_accuracy_DT,
            ]
            KNN_RESULTS = [
                confusion_matrix_knn,
                classification_report_knn,
                score_accuracy_KNN,
            ]
            SVM.append(SVM_RESULTS)
            KNN.append(KNN_RESULTS)
            DT.append(DT_RESULTS)
        return SVM, KNN, DT

    def CrossValidation(self, image, labels, test_size):
        """
        Perform cross-validation on the given image and labels dataset.

        Parameters:
            image (array-like): The input image dataset.
            labels (array-like): The corresponding labels for the image dataset.
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            tuple: A tuple containing two lists, X and Y. The list X contains the train and test splits
                   of the input image dataset. The list Y contains the train and test splits of the
                   corresponding labels dataset.
        """
        X = []
        Y = []
        X_train, X_test, y_train, y_test = train_test_split(
            image, labels, test_size=test_size
        )
        X.append(X_train)
        X.append(X_test)
        Y.append(y_train)
        Y.append(y_test)
        print()
        return X, Y

    def ROC_CURVE(self, label_test, label_score):
        """
        Generate a Receiver Operating Characteristic (ROC) curve for binary or multi-class classification.

        Parameters:
            label_test (ndarray): The true labels of the test data. The shape of the array should be (n_samples, n_classes).
            label_score (ndarray): The predicted scores or probabilities for the test data. The shape of the array should be (n_samples, n_classes).

        Returns:
            None

        This function plots the ROC curve for each class separately, as well as the micro-average ROC curve that summarizes the performance across all classes. The area under the ROC curve (AUC) is also computed for each class and the micro-average.

        Note:
            - The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds.
            - The micro-average ROC curve and AUC are computed by flattening the true labels and predicted scores arrays.
            - The function uses the `roc_curve` function from the `sklearn.metrics` module to compute the TPR and FPR for each class.
            - The function uses the `auc` function from the `sklearn.metrics` module to compute the AUC for each class.

        Example Usage:
            label_test = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
            label_score = np.array([[0.2, 0.8, 0.6], [0.7, 0.3, 0.5], [0.9, 0.4, 0.7]])
            ROC_CURVE(label_test, label_score)
        """
        n_classes = label_test.shape
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(label_test[:, i], label_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            label_test.ravel(), label_score.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr[2], tpr[2], label="ROC curve (area = %0.2f)" % roc_auc[2])
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
        )
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multi-class")
        plt.legend(loc="lower right")
        plt.show()
