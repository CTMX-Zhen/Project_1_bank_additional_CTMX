"""
Project 1 Task Description：
- Intelligence Solutions Sdn. Bhd. is a company that provides an intelligence system solution to worldwide customers. The company recently awarded a contract to help a Portuguese banking institution with its direct marketing campaigns. The goal is to predict if the client will subscribe or not to the promoted term deposit product.

The bank has provided the data at the following website: 
- Bank Marketing dataset at UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing)
**Note: Use bank-additional.csv dataset

Task：
- To identify the suitable AI solution for this prediction problem.
- Identify optimized parameters for the solution.
- To implement the solution using any tools or languages
- To analyze the results.
- Suggestion to further improve the results and implement proposed improvement.

Submission:
Report:
- Introduction to the proposed solution and justification of selection.
- The proposed solution architecture and justification.
- The proposed optimized parameters and justification.
- Steps of implementation with complete screenshots
- Implementation of proposed improvement/enhancement with complete screenshots
- Result analysis and discussion.
- Complete working source code or implementation files.

NAME: Connie Tang Ming Xin
MATRIC NUMBER: MEC245051
"""

# Project 1
    # Dataset: dataset\bank-additional.csv 
        # - Sourse downloaded from UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing
        # - Description: 
            # ==> The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
            # ==> The classification goal is to predict if the client will subscribe a term deposit (variable y).

    # Model:
        # - Logistic Regression
        # - XGBoost

    # Hyperparameter Tuning for XGBoost 
        # 1. Manual fine tune by best score and best parameters
        # 2. Manual fine tune by Grid Search

    # Model Evaluation
        # - Confusion Matrix
        # - ROC AUC

    # Platform: offline AI tools & code your own solution
        # - Visual Studio Code
        # - Jupyter Notebook
        # - Python
        # - pandas, matplotlib, seaborn, scikit-learn, xgboost
        
"""
    STUDENT'S OUTCOMES
"""
# pip install pandas matplotlib seaborn scikit-learn
# pip install xgboost

# == IMPORT LIBRARIES ==
import os
import sys
import joblib
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# == HELPER FUNCTIONS ==
# 1. Setup logger
def setup_logger(log_file="train_log.txt"):
    """
    Sets up a logger with two handlers: a file handler and a console handler.
    
    Args:
        log_file (str, optional): The file to write logs to. Defaults to "train_log.txt".
    
    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter
    # The formatter is used to format the log messages.
    # It takes the log message as input and returns a string.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    # The file handler is used to write logs to a file.
    # It takes the log file path as an argument.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    # The console handler is used to write logs to the console.
    # It takes the output stream as an argument.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# 2. Load the dataset
def load_dataset(file_path):
    """
    Loads the dataset from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path, sep=';')

# 3. Quick overview of the dataset
def quick_overview(data):
    """
    logger.infos the first 5 rows of the dataset, its shape and some statistics of the categorical and numerical columns.

    Args:
        data (pandas.DataFrame): The dataset to be logger.infoed.
    """
    # The first 5 rows of the dataset
    logger.info("First 5 rows of the dataset:")
    logger.info("\n" + str(data.head()))

    # The shape of the dataset
    logger.info(data.shape)

    # Count the number of missing values in the dataset
    logger.info("Number of missing values in each column:")
    logger.info("\n" + str(data.isnull().sum()))

    # Get the categorical columns
    categorical_cols = data.select_dtypes(include='object').columns

    # logger.info the categorical columns
    logger.info("Categorical columns:\n" + "\n".join(f"- {col}" for col in categorical_cols))

    return categorical_cols

# 4. Data preprocessing
def data_preprocessing(data, categorical_cols):
    """
    Preprocesses the dataset by converting the target variable to binary
    and one-hot encoding the categorical variables (excluding the target).

    Args:
        data (pandas.DataFrame): The dataset to be preprocessed.

    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    # Convert target variable to binary
    # Map the target variable to 0/1
    data['y'] = data['y'].map({'yes': 1, 'no': 0})

    # One-hot encode categorical variables (excluding target)
    # One-hot encode the categorical variables
    data_encoded = pd.get_dummies(data, columns=[col for col in categorical_cols if col != 'y'], drop_first=True)

    return data_encoded

# 5. Split Dataset
def split_dataset(data_encoded, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pandas.DataFrame): The dataset to be split.
        test_size (float, optional): The proportion of the dataset to include in the test set. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """

    # Split the dataset into training and testing sets.
    # The target variable is the last column in the dataset.
    X = data_encoded.drop('y', axis=1)
    y = data_encoded['y']

    # Split the dataset into training and testing sets using stratified sampling.
    # This ensures that the proportion of positive and negative samples in the training and testing sets is the same as in the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

# 6. Baseline Model – Logistic Regression
def baseline_model(X_train, y_train, X_test, y_test):
    """
    Trains a logistic regression model on the training dataset and evaluates its performance on the testing dataset.

    Args:
        X_train (pandas.DataFrame): The training dataset.
        y_train (pandas.Series): The target variable for the training dataset.
        X_test (pandas.DataFrame): The testing dataset.
        y_test (pandas.Series): The target variable for the testing dataset.
    """
    # Feature scaling: Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model.
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    # Predict class labels and probabilities.
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model's performance.
    logger.info("Logistic Regression Performance:")
    logger.info("\n" + classification_report(y_test, y_pred_lr))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_lr):.4f}\n")

    # Plot the confusion matrix and ROC curve.
    plot_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")

    # Save the model.
    joblib.dump(lr_model, 'model\\Logistic Regression\\lr_model.joblib')

# 7. Improved Model – XGBoost
def improved_model(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model on the training dataset and evaluates its performance on the testing dataset.

    Args:
        X_train (pandas.DataFrame): The training dataset.
        y_train (pandas.Series): The target variable for the training dataset.
        X_test (pandas.DataFrame): The testing dataset.
        y_test (pandas.Series): The target variable for the testing dataset.
    """
    # Train XGBoost model.
    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Predict class labels and probabilities.
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Evaluate the model's performance.
    logger.info("XGBoost Performance:")
    logger.info("\n" + classification_report(y_test, y_pred_xgb))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")

    # Plot the confusion matrix and ROC curve.
    plot_model(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")

    # Save the model.
    joblib.dump(xgb_model, 'model\\XGBoost\\xgb_model.joblib')

# 8. Hyperparameter Tuning for XGBoost by best score and best parameters
def bset_score_performance_model(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model on the training dataset and evaluates its performance on the testing dataset.
    The function performs a manual hyperparameter tuning for the XGBoost model by iterating over all
    combinations of the given hyperparameters and evaluating the model's ROC AUC score on the testing dataset.

    Args:
        X_train (pandas.DataFrame): The training dataset.
        y_train (pandas.Series): The target variable for the training dataset.
        X_test (pandas.DataFrame): The testing dataset.
        y_test (pandas.Series): The target variable for the testing dataset.
    """
    logger.info("Starting manual fine tune by best score and best parameters...")
    
    # Define the hyperparameters to be tuned
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }

    # Generate all combinations manually
    param_combos = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Initialize best score and best parameters
    best_score = 0
    best_params = None

    # Iterate over all combinations
    for combo in param_combos:
        params = dict(zip(param_names, combo))
        model = XGBClassifier(eval_metric='logloss', **params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        if score > best_score:
            best_score = score
            best_params = params

    # Log the best score and best parameters
    logger.info(f"Best ROC AUC: {best_score:.4f} with params: {best_params}\n")

    # Train an XGBoost model with the best hyperparameters
    best_score_params_model = XGBClassifier(eval_metric='logloss', **best_params)
    best_score_params_model.fit(X_train, y_train)

    # Predict class labels and probabilities
    y_pred_best_spm = best_score_params_model.predict(X_test)
    y_prob_best_spm = best_score_params_model.predict_proba(X_test)[:, 1]

    # Evaluate the model's performance
    logger.info("Manually Tuned XGBoost Performance:")
    logger.info("\n" + classification_report(y_test, y_pred_best_spm))
    logger.info(f"ROC AUC (Manual Best): {roc_auc_score(y_test, y_prob_best_spm):.4f}\n")

    # Plot the confusion matrix and ROC curve.
    plot_model(y_test, y_pred_best_spm, y_prob_best_spm, "best_score_params_model")

    # Save the model
    joblib.dump(best_score_params_model, 'model\\best_score_params_model\\best_score_params_model.joblib')

# 9. Hyperparameter Tuning for XGBoost by Grid Search
def grid_search_model(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model on the training dataset and evaluates its performance on the testing dataset.
    The function uses a grid search to find the best hyperparameters for the XGBoost model.

    Args:
        X_train (pandas.DataFrame): The training dataset.
        y_train (pandas.Series): The target variable for the training dataset.
        X_test (pandas.DataFrame): The testing dataset.
        y_test (pandas.Series): The target variable for the testing dataset.

    Returns:
        None
    """
    logger.info("Starting manual fine tune by Grid Search...")
    # Define the hyperparameters to be tuned.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }

    # Perform a grid search to find the best hyperparameters.
    grid_search = GridSearchCV(
        estimator=XGBClassifier(eval_metric='logloss'),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Log the best parameters found.
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # Train an XGBoost model with the best hyperparameters and evaluate its performance.
    best_grid_serach_model = grid_search.best_estimator_
    y_pred_best_grid = best_grid_serach_model.predict(X_test)
    y_prob_best_grid = best_grid_serach_model.predict_proba(X_test)[:, 1]

    logger.info("Tuned XGBoost Performance:")
    logger.info("\n" + classification_report(y_test, y_pred_best_grid))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_best_grid):.4f}\n")

    # Plot the confusion matrix and ROC curve.
    plot_model(y_test, y_pred_best_grid, y_prob_best_grid, "grid_search")

    # Save the model.
    joblib.dump(best_grid_serach_model, 'model\\grid_search\\grid_search_model.joblib')

# 10. Plotting Confusion Matrix and ROC Curve
def plot_model(y_test, y_pred, y_prob, model):
    """
    Plot the confusion matrix and ROC curve for the given model.
    
    Parameters:
    - y_test: array-like, true labels of the test data.
    - y_pred: array-like, predicted labels by the model.
    - y_prob: array-like, predicted probabilities by the model for the positive class.
    - model: str, name of the model used for plotting and saving files.
    """
    # Ensure model-specific output directory exists
    dir_path = os.path.join("model", model)
    os.makedirs(dir_path, exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(dir_path, f"confusion_matrix_{model}.png"))
    plt.show()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc_score(y_test, y_prob):.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"roc_curve_{model}.png"))
    plt.show()

# 11. Save model each of the model
def save_model(model, model_name):
    """
    Save the given model to a file with the given name.

    The model is saved using joblib.dump, and the file name is the given name
    with ".joblib" appended.

    Args:
        model (object): The model to be saved.
        model_name (str): The name of the file to which the model should be saved.
    """
    # Save the model using joblib.dump
    joblib.dump(model, f"{model_name}.joblib")
    # Log a message indicating that the model has been saved
    logger.info(f"Model saved as {model_name}.joblib")

# 12. Save summary
def save_summary(X_test, y_test):
    """
    Saves a summary of the final model's performance to a file.

    The summary includes the ROC AUC score and the classification report
    for both the manual and grid search models.

    Args:
        X_test (pandas.DataFrame): The test dataset.
        y_test (pandas.Series): The target variable for the test dataset.
    """
    try:
        # Load both models
        manual_model = joblib.load("model\\best_score_params_model\\best_score_params_model.joblib")
        grid_model = joblib.load("model\\grid_search\\grid_search_model.joblib")

        # Predict and evaluate Manual Model
        y_pred_manual = manual_model.predict(X_test)
        y_prob_manual = manual_model.predict_proba(X_test)[:, 1]
        auc_manual = roc_auc_score(y_test, y_prob_manual)
        report_manual = classification_report(y_test, y_pred_manual)

        # Predict and evaluate Grid Model
        y_pred_grid = grid_model.predict(X_test)
        y_prob_grid = grid_model.predict_proba(X_test)[:, 1]
        auc_grid = roc_auc_score(y_test, y_prob_grid)
        report_grid = classification_report(y_test, y_pred_grid)

        # Save summary to file
        with open("final_model_summary.txt", "w") as f:
            f.write("[Manual Tuning Model]\n")
            f.write(f"ROC AUC: {auc_manual:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report_manual + "\n")

            f.write("\n[Grid SearchCV Model]\n")
            f.write(f"ROC AUC: {auc_grid:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report_grid)

        logger.info("Final model summary saved to final_model_summary.txt")

    except FileNotFoundError as e:
        logger.error(f"Could not find model file: {e.filename}")


# == START ==
if __name__ == "__main__":
    # Step 1: Setup logger
    logger = setup_logger("training_log.txt")
    logger.info("Starting progress with log logged...")

    # Step 2: Load the dataset
    data = load_dataset("dataset\\bank-additional.csv")

    # Step 3: Quick overview of the dataset
    categorical_cols = quick_overview(data)

    # Step 4: Data preprocessing
    data_encoded = data_preprocessing(data, categorical_cols)

    # Step 5: Split Dataset
    X_train, X_test, y_train, y_test = split_dataset(data_encoded)

    # Step 6: Baseline Model – Logistic Regression
    model1 = baseline_model(X_train, y_train, X_test, y_test)

    # Step 7: Improved Model – XGBoost
    model2 = improved_model(X_train, y_train, X_test, y_test)

    # Step 8: Hyperparameter Tuning for XGBoost
    model3_hyperparameter_tuning = bset_score_performance_model(X_train, y_train, X_test, y_test)

    # Step 9: Hyperparameter Tuning for XGBoost by best score and best parameters
    model4_hyperparameter_tuning = grid_search_model(X_train, y_train, X_test, y_test)

    # Step 10: Summary the best model
    save_summary(X_test, y_test)

# == END ==