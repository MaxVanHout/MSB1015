import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, BaseCrossValidator
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator
import random
import string


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Identifies features containing missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.
    
    Returns:
        pd.DataFrame or None: A DataFrame listing features with the number of missing values.
                             Returns None if no missing values are found.
    """
    # Count the number of missing values in each column of the DataFrame
    missing_count = df.isna().sum()
    # Calculate the percentage of missing values for each column
    missing_percentage = (missing_count / len(df)) * 100
    # Select only those columns that have missing values (count > 0)
    missing_values = missing_count[missing_count > 0]
    # Select the corresponding percentages of missing values for the columns with missing data
    missing_percentages = missing_percentage[missing_count > 0]
    # Check if there are any columns with missing values
    if not missing_values.empty:
        # Create a DataFrame to summarize the missing value information
        result = pd.DataFrame({
            'Feature': missing_values.index,                         
            'Number of Missing Values': missing_values.values,      
            'Percentage of Missing Values (%)': missing_percentages.values  
        })
        return result  # Return the DataFrame containing the missing values summary
    else:
        return None  # Return None if there are no missing values in the DataFrame
    

def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Identifies duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to check for duplicates.
    
    Returns:
        pd.DataFrame or None: A DataFrame containing the duplicate rows.
                             Returns None if no duplicates are found.
    """
    # Return the duplicate rows if there are any, otherwise none
    return df[df.duplicated()] if df.duplicated().any() else None


def get_non_numeric_values(features_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Identifies non-numeric entries in each feature of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the feature data to check for non-numeric entries.
    
    Returns:
        pd.DataFrame or None: A DataFrame listing the features, indices, and non-numeric values found.
                             Returns None if no non-numeric values are found.
    """
    # Attempt to convert all values in the DataFrame to numeric types
    # Any value that cannot be converted will result in NaN
    non_numeric = features_df.apply(pd.to_numeric, errors='coerce').isna()
    # Create a list to store details of non-numeric values
    non_numeric_list = [
        {
            'Feature': col,  # The name of the feature (column) with a non-numeric value
            'Index': idx,    # The index of the row where the non-numeric value is located
            'Non Numeric Value': features_df.at[idx, col]  # The actual non-numeric value found
        }
        for col in non_numeric.columns                       # Loop over each column in the DataFrame
        for idx in non_numeric.index[non_numeric[col]]     # Loop over each index where the value is NaN
    ]
    # Return a DataFrame summarizing the non-numeric values if any were found; otherwise, return None
    return pd.DataFrame(non_numeric_list) if non_numeric_list else None


def get_inconsistent_decimal_separator(features_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Identifies inconsistent decimal separators (commas) in numeric columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to check for inconsistent decimal separators.
    
    Returns:
        pd.DataFrame or None: A DataFrame listing features, indices, and inconsistent values with commas.
                             Returns None if no inconsistencies are found.
    """
    # Create a list to store details of values containing commas
    inconsistent_list = [
        {
            'Feature': col,  # The name of the feature (column) with an inconsistent value
            'Index': idx,    # The index of the row where the inconsistent value is located
            'Inconsistent Value': features_df.at[idx, col]  # The actual inconsistent value found
        }
        for col in features_df.columns  # Loop over each column in the DataFrame
        # Find the indices of rows where the value in the column contains a comma (inconsistent)
        for idx in features_df[features_df[col].astype(str).str.contains(',', na=False)].index 
    ]
    # Return a DataFrame summarizing the inconsistent values if any were found; otherwise, return None
    return pd.DataFrame(inconsistent_list) if inconsistent_list else None


def get_unique_target_values(target_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Returns the unique values found in the target variable DataFrame
    
    Args:
        df (pd.DataFrame): The DataFrame to check for unique values.
    
    Returns:
        pd.DataFrame or None: A DataFrame listing features and their unique values.
                             Returns None if no non-numeric columns are present.
    """
    # Create a DataFrame to store unique values for each target feature
    unique_values_df = pd.DataFrame({
        'Target': target_df.columns,  # Column names (target features)
        'Unique Entries': [target_df[col].unique().tolist() for col in target_df]  # Unique values for each column as a list
    })
    # Return the DataFrame containing unique values if it's not empty; otherwise, return None
    return unique_values_df if not unique_values_df.empty else None


def check_data_quality(df: pd.DataFrame, target_col: str) -> dict:
    """
    Splits the input DataFrame into features and target, and performs various data quality checks on the features. 
    The function checks for missing values, duplicate rows, non-numeric entries, unique values in the target, and inconsistent decimal separators.

    Args:
        df (pd.DataFrame): The DataFrame containing both feature columns and the target column.
        target_col (str): The name of the column that represents the target variable, which will be separated from the feature columns.

    Returns:
        dict: A dictionary with the names of each check as keys and the corresponding result DataFrame as values. 
              If no issues are found in a check, that check is excluded from the dictionary.
    """
    # Separate features and target from the original DataFrame
    features_df = df.drop(columns=[target_col])  # Drop the target column to keep only feature columns
    target_df = pd.DataFrame(df[target_col])      # Create a DataFrame for the target column
    # Run various data quality checks and store the results in a dictionary
    checks = {
        "Missing Values": get_missing_values(df),                       # Check for missing values in the DataFrame
        "Duplicate Rows": get_duplicate_rows(df),                       # Check for any duplicate rows in the DataFrame
        "Non-Numeric Entries": get_non_numeric_values(features_df),     # Check for non-numeric entries in feature columns
        "Classes": get_unique_target_values(target_df),                 # Check for unique class values in the target column
        "Inconsistent Decimal Separators": get_inconsistent_decimal_separator(features_df),  # Check for inconsistent decimal separators in features
    }
    # Iterate through each check and its result
    for check, result in checks.items():
        if result is None:
            # Print a message if no issues were found for this check
            print(f"No {check.lower()} found.\n")
        else:
            # If issues were found, print the results accordingly
            if check == 'Classes':
                # For classes, print the unique entries found
                print(f"{check} found: {result['Unique Entries'].tolist()}\n")
            else:
                # For other checks, print the number of issues found
                print(f"{check} found: {result.shape[0]}\n")
    # Return a dictionary of results for checks that found issues, excluding those that are None
    return {check: result for check, result in checks.items() if result is not None}


def calculate_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various measures of spread for numerical features in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the numerical feature data. 

    Returns:
        pd.DataFrame: A DataFrame summarizing various spread metrics for each numerical feature.
    """
    # Select only the numeric columns from the DataFrame
    df = df.select_dtypes(include='number')
    # Calculate the range (max - min) for each numeric feature
    range_values = df.max() - df.min()
    # Calculate the first quartile (25th percentile) for each feature
    Q1 = df.quantile(0.25)
    # Calculate the third quartile (75th percentile) for each feature
    Q3 = df.quantile(0.75)
    # Calculate the Interquartile Range (IQR) for each feature
    IQR = Q3 - Q1
    # Calculate the mean for each feature
    means = df.mean()
    # Calculate the standard deviation for each feature
    SD = df.std()
    # Calculate the Median Absolute Deviation (MAD) for each feature
    MAD = abs(df - df.median()).median()
    # Count the number of zeros in each feature
    zero_count = (df == 0).sum()
    # Calculate the percentage of zeros for each feature
    zero_percentage = (((df == 0).sum()) / len(df)) * 100
    # Create a DataFrame to summarize the statistics for each feature
    feature_stats = pd.DataFrame({
        'Feature': df.columns,                    
        'Mean': means,                            
        'Range': range_values,                    
        'IQR': IQR,                              
        'MAD': MAD,                              
        'Variance': df.var(),                    
        'SD': SD,                                
        'Number of Zeros': zero_count,           
        'Percentage of Zeros': zero_percentage   
    }).reset_index(drop=True)  # Reset index to have a clean DataFrame without an index column

    return feature_stats  # Return the DataFrame with feature statistics


def feature_selection_filter(X: pd.DataFrame, 
                              y: pd.Series | np.ndarray, 
                              model: BaseEstimator, 
                              cv: BaseCrossValidator, 
                              score_func: callable) -> pd.DataFrame:
    """
    Performs feature selection using SelectKBest with the provided scoring function and evaluates
    the performance of the model using cross-validation for a range of features (2 to 20).

    Parameters:
    ----------
    X : pandas.DataFrame
        Input features (predictors).
        
    y : pandas.Series or numpy.ndarray
        Target variable (labels).
        
    model : sklearn estimator
        The machine learning model to be evaluated.
        
    cv : cross-validation generator
        Cross-validation strategy, such as StratifiedKFold.
        
    score_func : callable
        Scoring function for feature selection (e.g., f_classif, mutual_info_classif).
        
    Returns:
    -------
    results_df : pandas.DataFrame
        A DataFrame containing the number of features, cross-validation mean accuracy, 
        cross-validation standard deviation, and selected feature names for each iteration.
    """
    # Initialize a list to store results
    results = []

    # Iterate over the number of features to select (from 2 to 20)
    for n_features in range(2, 21):
        # Use SelectKBest to select the top n features using the specified scoring function
        selector = SelectKBest(score_func=score_func, k=n_features)
        X_selected = selector.fit_transform(X, y)

        # Get the names of the selected features
        selected_feature_names = X.columns[selector.get_support()].tolist()

        # Standardize the selected features
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X_selected)

        # Perform cross-validation and calculate accuracy
        cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')

        # Store the results (mean and std of cross-validation scores)
        results.append({
            'n_features': n_features,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'selected_features': selected_feature_names
        })

        # Print the cross-validation results for this number of features
        print(f"Number of Features: {n_features}, CV Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Find the optimal number of features with the highest CV mean accuracy
    best_result = results_df.loc[results_df['cv_mean_accuracy'].idxmax()]
    print(f"\nOptimal Number of Features: {best_result['n_features']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f} ± {best_result['cv_std_accuracy']:.4f}")
    
    return results_df


def recursive_feature_elimination(X: pd.DataFrame, 
                                   y: pd.Series | np.ndarray, 
                                   estimator: BaseEstimator, 
                                   cv: BaseCrossValidator, 
                                   param_grid: dict) -> tuple[pd.DataFrame, BaseEstimator]:
    """
    Perform Recursive Feature Elimination (RFE) in combination with Grid Search 
    for hyperparameter tuning to select the optimal number of features for a given estimator.
    
    Parameters:
    ----------
    X : pandas.DataFrame
        The input features (predictors) of the dataset.
        
    y : pandas.Series or numpy.ndarray
        The target labels corresponding to X.
        
    estimator : object
        The machine learning estimator (e.g., decision tree, random forest) that will be used 
        for feature selection and model fitting. The estimator must have a `fit` method.
        
    cv : int or cross-validation generator
        The cross-validation strategy to use for model evaluation. This can be an integer specifying 
        the number of folds (for KFold cross-validation) or a cross-validation generator object.
        
    param_grid : dict
        The hyperparameter grid for the estimator, used in GridSearchCV for hyperparameter tuning. 
        The grid should contain parameters that are applicable to the estimator.
    
    Returns:
    -------
    results_df : pandas.DataFrame
        A DataFrame containing results for each iteration, including:
        - 'n_features': The number of features selected.
        - 'cv_mean_accuracy': The mean cross-validated accuracy for the model.
        - 'cv_std_accuracy': The standard deviation of cross-validated accuracy.
        - 'best_params': The best hyperparameters found during grid search.
        - 'selected_features': The list of selected feature names for that iteration.
        
    best_model : BaseEstimator
        The best estimator (e.g., decision tree) fitted with the optimal hyperparameters and features.
    """
    results = []
    best_model = None
    best_cv_mean_accuracy = 0

    # Loop through different numbers of selected features
    for n_features in range(2, 21):
        print(f"Analyzing RFE with {n_features} features")
        # Recursive Feature Elimination (RFE) to select n_features
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)

        # Grid Search for hyperparameter tuning
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_selected, y)

        # Best model from grid search
        best_model_for_iteration = grid_search.best_estimator_
        # Cross-validation with the selected features
        cv_scores = cross_val_score(best_model_for_iteration, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)

        # Get the feature selection mask and selected feature names
        selected_feature_names = X.columns[rfe.support_].tolist()

        # Append the results (mean, std accuracy, best params, and selected features)
        results.append({
            'n_features': n_features,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'best_params': grid_search.best_params_,
            'selected_features': selected_feature_names
        })
        # Check if the current model is better than the previous best model and store in best_model if true
        if np.mean(cv_scores) > best_cv_mean_accuracy:
            best_cv_mean_accuracy = np.mean(cv_scores)
            best_model = best_model_for_iteration
        # Print cross-validation results for this number of features
        print(f"Number of Features: {n_features}, CV Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    results_df = pd.DataFrame(results)
    # Find the best result based on cross-validated mean accuracy and print the result
    best_result = results_df.loc[results_df['cv_mean_accuracy'].idxmax()]
    print(f"\nOptimal Number of Features: {best_result['n_features']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f} ± {best_result['cv_std_accuracy']:.4f}")

    return results_df, best_model


def corrupt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce missing values, duplicate rows, non-numerical entries,
    and misspelled strings into a DataFrame with randomly generated fractions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be corrupted.

    Returns:
    - pd.DataFrame: The corrupted DataFrame.
    """
   # Introduce missing values into the DataFrame
    num_missing = int(0.01 * df.size)  # Calculate the number of missing values (1% of total size)
    missing_indices = random.sample(range(df.size), num_missing)  # Randomly select indices for missing values
    # Set the selected indices to NaN (missing values)
    for idx in missing_indices:
        df.iat[idx // df.shape[1], idx % df.shape[1]] = np.nan  # Use integer division and modulus to locate the row and column

    # Introduce non-numerical entries in numerical columns
    for col in df.select_dtypes(include=np.number).columns:  # Loop through each numerical column
        num_non_numeric = int(0.0001 * len(df))  # Calculate number of non-numeric entries (0.01% of total rows)
        non_numeric_indices = random.sample(range(len(df)), num_non_numeric)  # Randomly select indices for non-numeric values

        # Replace selected indices with random non-numeric entries
        for idx in non_numeric_indices:
            df.at[idx, col] = random.choice(['a', 'b', 'c', 'invalid', 'non-numeric'])  # Replace with random non-numeric value

    # Introduce misspellings in string columns
    for col in df.select_dtypes(include='object').columns:  # Loop through each string column
        num_misspellings = int(0.01 * len(df))  # Calculate number of misspellings (1% of total rows)
        misspell_indices = random.sample(range(len(df)), num_misspellings)  # Randomly select indices for misspellings

        # Replace the original value with a misspelled version
        for idx in misspell_indices:
            original_value = df.at[idx, col]  # Get the original value
            if isinstance(original_value, str):  # Check if the original value is a string
                # Randomly generate a misspelled version of the string
                misspelled_value = ''.join(random.choice(original_value) for _ in range(len(original_value)))  # Generate a misspelled string
                df.at[idx, col] = misspelled_value  # Replace the original value with the misspelled value

    # Introduce duplicate rows
    num_rows_to_duplicate = int(0.05 * len(df))  # Calculate the number of rows to duplicate (5% of total rows)
    if num_rows_to_duplicate > 0:  # Check if there are rows to duplicate
        duplicates = df.sample(num_rows_to_duplicate, replace=True)  # Randomly sample rows to duplicate
        df = pd.concat([df, duplicates], ignore_index=True)  # Concatenate duplicates with the original DataFrame

    return df  # Return the modified DataFrame with introduced data issues