
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score


def main():
    # read data
    data = pd.read_csv('data/bank_full.csv')
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.expand_frame_repr', False)

    # check data
    print('======Original Data======')
    # print(data.isnull().sum())  # check if there are missing values
    print(data.shape)  # check the data amount

    # using plot to check if data is imbalanced
    count_y = data['y'].value_counts()
    print(count_y)
    # plt.figure(figsize=(5, 5))
    # data['y'].value_counts().plot(kind='pie', colors = ['lightcoral', 'skyblue'], autopct='%1.2f%%')
    # plt.title('Data Distribution')
    # plt.ylabel('')
    # plt.show()

    # data preprocessing
    columns = data.columns.tolist()
    columns = [c for c in columns if c not in ['y']]
    for col in columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    target = 'y'
    data[target] = LabelEncoder().fit_transform(data[target])

    x = data[columns]
    y = data[target]

    # split data
    # use 80% of train data and 20% of test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # over sampling
    print('======Over Sampling======')
    smote = SMOTE()
    x_train_os, y_train_os = smote.fit_resample(x_train, y_train)

    print(x_train_os.shape)
    print(y_train_os.shape)

    count_os = y_train_os.value_counts()
    print(count_os)

    # plt.figure(figsize=(5, 5))
    # y_train_os.value_counts().plot(kind='pie', colors = ['lightcoral', 'skyblue'], autopct='%1.2f%%')
    # plt.title('Over Sampling Data Distribution')
    # plt.ylabel('')
    # plt.show()

    # modeling
    feature_names = x_train.columns

    # logistic regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train_os, y_train_os)

    feature_coefficients_lr = logistic_regression.coef_[0]
    feature_coefficient_pairs_lr = list(zip(feature_names, feature_coefficients_lr))
    feature_coefficient_pairs_lr.sort(key=lambda x: abs(x[1]), reverse=True)

    evaluate_model("Logistic Regression", logistic_regression, x_test, y_test, logistic_regression.predict_proba(x_test)[:, 1], feature_coefficient_pairs_lr)

    # decision tree
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.fit(x_train_os, y_train_os)

    feature_importances_dt = decision_tree.feature_importances_
    feature_importance_pairs_dt = list(zip(feature_names, feature_importances_dt))
    feature_importance_pairs_dt.sort(key=lambda x: x[1], reverse=True)

    evaluate_model("Decision Tree", decision_tree, x_test, y_test, decision_tree.predict_proba(x_test)[:, 1], feature_importance_pairs_dt)

    # random forest
    random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
    random_forest.fit(x_train_os, y_train_os)

    feature_importances_rf = random_forest.feature_importances_
    feature_importance_pairs_rf = list(zip(feature_names, feature_importances_rf))
    feature_importance_pairs_rf.sort(key=lambda x: x[1], reverse=True)

    evaluate_model("Random Forest", random_forest, x_test, y_test, random_forest.predict_proba(x_test)[:, 1], feature_importance_pairs_rf)

    # gradient boosting
    gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=0)
    gradient_boosting.fit(x_train_os, y_train_os)

    feature_importances_gb = gradient_boosting.feature_importances_
    feature_importance_pairs_gb = list(zip(feature_names, feature_importances_gb))
    feature_importance_pairs_gb.sort(key=lambda x: x[1], reverse=True)

    evaluate_model("Gradient Boosting", gradient_boosting, x_test, y_test, gradient_boosting.predict_proba(x_test)[:, 1], feature_importance_pairs_gb)


def evaluate_model(model_name, model, x_test, y_test, y_proba, feature_importance_pairs=None):
    print(f"======{model_name}======")

    if feature_importance_pairs:
        print(f"Feature Importance for {model_name}:")
        for feature, importance in feature_importance_pairs:
            print(f"Feature: {feature}, Importance: {importance}")

    # predictions
    y_pred = model.predict(x_test)

    # confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", roc_auc)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # recall
    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)

    # # compute ROC curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    #
    # # plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Receiver Operating Characteristic (ROC) for {model_name}')
    # plt.legend(loc='lower right')
    # plt.show()












if __name__ == '__main__':
    main()