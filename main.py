import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score
import random
from GRANDE import GRANDE
from sklearn.model_selection import train_test_split

from data import load_data, sample

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Model Training
def Train(directory, times):
    # Store metrics for each time.
    times_precision = np.array([])
    times_recall = np.array([])
    times_accuracy = np.array([])
    times_f1score = np.array([])
    times_auc = np.array([])
    times_aupr = np.array([])

    for time in range(times):
        print(
            f'------------------------------------------ Times {time + 1} Begin ------------------------------------------')
        # Read lncRNA and disease features.
        lncRNA_feature, disease_feature = load_data(directory)
        # Selection of positive and negative samples
        samples = sample(directory, int(random.random()))

        # Data Preparation [lncRNA_disease, disease_feature, label]
        dataset = []
        for k in range(samples.shape[0]):
            lncRNA_index = samples[k, 0] - 1
            disease_index = samples[k, 1] - 1
            label = samples[k, 2]
            dataset.append(np.hstack((lncRNA_feature[lncRNA_index], disease_feature[disease_index], label)))

        # Randomly sample and shuffle the order of samples (frac=1 indicates sampling all samples)
        all_dataset = pd.DataFrame(dataset)
        all_dataset = all_dataset.sample(frac=1).values

        # Sample features [lncRNA_feature, disease_feature]
        feature = all_dataset[:, :-1]
        # Sample labels [label]
        label = all_dataset[:, -1].reshape(-1)

        # Store metrics for each fold of training.
        cv_precision = np.array([])
        cv_recall = np.array([])
        cv_accuracy = np.array([])
        cv_f1score = np.array([])
        cv_auc = np.array([])
        cv_aupr = np.array([])

        # Cross-validation flag for each fold
        CV_flag = 1
        # 5-fold cross-validation
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(feature, label):
            train_feature = feature[train_index]
            test_feature = feature[test_index]
            train_label = label[train_index]
            test_label = label[test_index]
            X_tr, X_va, y_tr, y_va = train_test_split(train_feature, train_label, test_size=0.25)

            # GRANDE algorithm parameters
            categorical_feature_indices = []
            params = {
                'depth': 5,
                'n_estimators': 2048,

                'learning_rate_weights': 0.005,
                'learning_rate_index': 0.01,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.01,

                'optimizer': 'SWA',
                'cosine_decay_steps': 0,

                'initializer': 'RandomNormal',

                'loss': 'crossentropy',
                'focal_loss': False,
                'temperature': 0.0,

                'from_logits': True,
                'apply_class_balancing': True,

                'dropout': 0.0,

                'selected_variables': 0.8,
                'data_subset_fraction': 1.0,
            }

            args = {
                'epochs': 150,
                'early_stopping_epochs': 25,
                'batch_size': 64,
                'cat_idx': categorical_feature_indices,  # put list of categorical indices
                'objective': 'binary',

                'metrics': ['F1'],  # F1, Accuracy, R2
                'random_seed': 42,
                'verbose': 1,
            }

            # Model training
            model = GRANDE(params=params, args=args)
            model.fit(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)

            # Test set score
            scores = model.predict(test_feature)[:, -1]
            scores_arr = np.arange(0, dtype=float)
            # Store scores
            for score in scores:
                scores_arr = np.append(scores_arr, score)
            # Convert scores to 0 and 1
            result_label = []
            for score in scores_arr:  # 0 1
                if score > 0.5:
                    result_label.append(1)
                else:
                    result_label.append(0)

            # Results for each fold of cross-validation
            precision = precision_score(test_label, result_label)
            recall = recall_score(test_label, result_label)
            accuracy = accuracy_score(test_label, result_label)
            f1score = f1_score(test_label, result_label)
            fpr, tpr, thresholds = roc_curve(test_label, scores_arr)
            pre, rec, thresholds1 = precision_recall_curve(test_label, scores_arr)
            au = auc(fpr, tpr)
            aupr = auc(rec, pre)

            # ROC curve data
            curve_ROC = np.vstack([fpr, tpr])
            curve_ROC = pd.DataFrame(curve_ROC.T)
            curve_ROC.to_csv(directory + 'curve/' + 'ROC_' + 'Time' + str(time) + '_' + 'CV' + str(CV_flag) + '.csv'
                             , header=None, index=None)

            # PR curve data
            curve_PR = np.vstack([rec, pre])
            curve_PR = pd.DataFrame(curve_PR.T)
            curve_PR.to_csv(directory + 'curve/' + 'PR_' + 'Time' + str(time) + '_' + 'CV' + str(CV_flag) + '.csv'
                            , header=None, index=None)

            # Print results for each fold
            print(
                f'------------------------------------------ Times {time + 1} CV {CV_flag} Result ------------------------------------------')
            CV_flag = CV_flag + 1
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'Accuracy: {accuracy}')
            print(f'F1-score: {f1score}')
            print(f'AUC: {au}')
            print(f'AUPR: {aupr}')

            # Append the results for each fold to an array
            cv_precision = np.append(cv_precision, precision)
            cv_recall = np.append(cv_recall, recall)
            cv_accuracy = np.append(cv_accuracy, accuracy)
            cv_f1score = np.append(cv_f1score, f1score)
            cv_auc = np.append(cv_auc, au)
            cv_aupr = np.append(cv_aupr, aupr)

        # Calculate the mean and standard deviation of metrics after 5-fold.
        cv_precision_mean = cv_precision.mean()
        cv_recall_mean = cv_recall.mean()
        cv_accuracy_mean = cv_accuracy.mean()
        cv_f1score_mean = cv_f1score.mean()
        cv_auc_mean = cv_auc.mean()
        cv_aupr_mean = cv_aupr.mean()

        cv_precision_std = np.std(cv_precision)
        cv_recall_std = np.std(cv_recall)
        cv_accuracy_std = np.std(cv_accuracy)
        cv_f1score_std = np.std(cv_f1score)
        cv_auc_std = np.std(cv_auc)
        cv_aupr_std = np.std(cv_aupr)

        # Print the results after each 5-fold.
        print(
            f'-------------------------------------------- Times {time + 1} Result -----------------------------------------')
        print("Precsion: {} ± {}".format(round(cv_precision_mean, 4), round(cv_precision_std, 4)))
        print("Recall: {} ± {}".format(round(cv_recall_mean, 4), round(cv_recall_std, 4)))
        print("Accuracy: {} ± {}".format(round(cv_accuracy_mean, 4), round(cv_accuracy_std, 4)))
        print("F1score: {} ± {}".format(round(cv_f1score_mean, 4), round(cv_f1score_std, 4)))
        print('AUC: {} ± {}'.format(round(cv_auc_mean, 4), round(cv_auc_std, 4)))
        print('AUPR: {} ± {}'.format(round(cv_aupr_mean, 4), round(cv_aupr_std, 4)))

        # Store the results after each 5-fold.
        times_precision = np.append(times_precision, cv_precision_mean)
        times_recall = np.append(times_recall, cv_recall_mean)
        times_accuracy = np.append(times_accuracy, cv_accuracy_mean)
        times_f1score = np.append(times_f1score, cv_f1score_mean)
        times_auc = np.append(times_auc, cv_auc_mean)
        times_aupr = np.append(times_aupr, cv_aupr_mean)

    # Calculate the mean and standard deviation over a certain number of times.
    times_precision_mean = times_precision.mean()
    times_recall_mean = times_recall.mean()
    times_accuracy_mean = times_accuracy.mean()
    times_f1score_mean = times_f1score.mean()
    times_auc_mean = times_auc.mean()
    times_aupr_mean = times_aupr.mean()

    times_precision_std = np.std(times_precision)
    times_recall_std = np.std(times_recall)
    times_accuracy_std = np.std(times_accuracy)
    times_f1score_std = np.std(times_f1score)
    times_auc_std = np.std(times_auc)
    times_aupr_std = np.std(times_aupr)

    # Print the final results after performing 5-fold cross-validation for a certain number of times.
    print(
        f'============================================ {times} Times CV Result =========================================')
    print("Precision: {} ± {}".format(round(times_precision_mean, 4), round(times_precision_std, 4)))
    print("Recall: {} ± {}".format(round(times_recall_mean, 4), round(times_recall_std, 4)))
    print("Accuracy: {} ± {}".format(round(times_accuracy_mean, 4), round(times_accuracy_std, 4)))
    print("F1score: {} ± {}".format(round(times_f1score_mean, 4), round(times_f1score_std, 4)))
    print('AUC: {} ± {}'.format(round(times_auc_mean, 4), round(times_auc_std, 4)))
    print('AUPR: {} ± {}'.format(round(times_aupr_mean, 4), round(times_aupr_std, 4)))


if __name__ == '__main__':
    directory = './data1/'
    times = 20
    Train(directory, times)
