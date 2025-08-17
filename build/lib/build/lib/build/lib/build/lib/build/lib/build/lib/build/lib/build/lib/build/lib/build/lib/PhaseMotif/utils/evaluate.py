from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_classifiers(classifier_list, figure_name, figure_type='ROC'):
    plt.figure()
    for i, (y_true, y_pred, classifier_name) in enumerate(classifier_list):
        # Calculate evaluation metrics
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        # fpr, tpr, _ = precision_recall_curve(y_true, y_pred)
        # 计算每个类别的权重
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        class_weights = np.zeros_like(y_true, dtype=float)
        class_weights[y_true == 0] = 1.0 / np.sum(y_true == 0)
        class_weights[y_true == 1] = 1.0 / np.sum(y_true == 1)

        # 计算加权的精度和召回率
        precision1, recall1, _ = precision_recall_curve(y_true, y_pred)

        # y_pred = [1 if prob > 0.5 else 0 for prob in y_pred]
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        precision = precision_score(y_true, y_pred > 0.5)
        AP = average_precision_score(y_true, y_pred, pos_label=1, sample_weight=None)
        recall = recall_score(y_true, y_pred > 0.5)
        f1 = f1_score(y_true, y_pred > 0.5)

        # Calculate ROC curve and AUC
        # fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall1, precision1)

        # Print evaluation metrics
        print(f'Classifier {classifier_name}:')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'ROC AUC: {roc_auc}')
        print(f'PR AUC: {pr_auc}')
        print(f'AP: {AP}')
        print('\n')

        # Plot ROC curve
        if figure_type == 'ROC':
            plt.plot(fpr, tpr, label=f'ROC curve of {classifier_name} (area = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
        elif figure_type == 'PR':
            # Plot PR curve
            plt.plot(recall1, precision1, label=f'PR curve of {classifier_name} (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        else:
            raise ValueError('Invalid figure type')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('The result of the classifier')
    plt.legend(loc="lower right")
    plt.savefig(f'draw_result/{figure_type}曲线of{figure_name}.png')
    plt.show()