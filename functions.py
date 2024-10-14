import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
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
    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_values = missing_count[missing_count > 0]
    missing_percentages = missing_percentage[missing_count > 0]
    if not missing_values.empty:
        result = pd.DataFrame({
            'Feature': missing_values.index,
            'Number of Missing Values': missing_values.values,
            'Percentage of Missing Values (%)': missing_percentages.values
        })
        return result
    else:
        return None
    

def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Identifies duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to check for duplicates.
    
    Returns:
        pd.DataFrame or None: A DataFrame containing the duplicate rows.
                             Returns None if no duplicates are found.
    """
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
    non_numeric = features_df.apply(pd.to_numeric, errors='coerce').isna()
    non_numeric_list = [
        {'Feature': col, 'Index': idx, 'Non Numeric Value': features_df.at[idx, col]}
        for col in non_numeric.columns for idx in non_numeric.index[non_numeric[col]]
    ]
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
    inconsistent_list = [
        {'Feature': col, 'Index': idx, 'Inconsistent Value': features_df.at[idx, col]}
        for col in features_df.columns for idx in features_df[features_df[col].astype(str).str.contains(',', na=False)].index
    ]
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
    unique_values_df = pd.DataFrame({
        'Target': target_df.columns,
        'Unique Entries': [target_df[col].unique().tolist() for col in target_df]
    })
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
    features_df = df.drop(columns=[target_col])
    target_df = pd.DataFrame(df[target_col])
    checks = {
        "Missing Values": get_missing_values(df),
        "Duplicate Rows": get_duplicate_rows(df),
        "Non-Numeric Entries": get_non_numeric_values(features_df),
        "Classes": get_unique_target_values(target_df),
        "Inconsistent Decimal Separators": get_inconsistent_decimal_separator(features_df),
    }
    for check, result in checks.items():
        if result is None:
            print(f"No {check.lower()} found.\n")
        else:
            if check == 'Classes':
                print(f"{check} found: {result['Unique Entries'].tolist()}\n")
            else:
                print(f"{check} found: {result.shape[0]}\n")
    
    return {check: result for check, result in checks.items() if result is not None}


def calculate_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various measures of spread for numerical features in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the numerical feature data. 

    Returns:
        pd.DataFrame: A DataFrame summarizing various spread metrics for each numerical feature.
    """
    df = df.select_dtypes(include='number')
    range_values = df.max() - df.min()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    means = df.mean()
    SD = df.std()
    MAD = abs(df - df.median()).median()
    zero_count = (df == 0).sum()
    zero_percentage = (((df == 0).sum())/len(df))*100

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
    }).reset_index(drop=True)

    return feature_stats


def recursive_feature_elimination(X, y, estimator, cv, param_grid):
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
        
    X_filtered : pandas.DataFrame
        The input dataset X reduced to the best subset of selected features, based on the optimal 
        number of features that yielded the highest cross-validated accuracy.
    """
    # Store results for each iteration
    results = []

    # Loop through different numbers of selected features (2 to 20)
    for n_features in range(2, 21):
        print(f"Analyzing RFE with {n_features} features")

        # Recursive Feature Elimination (RFE) to select n_features
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)

        # Grid Search for hyperparameter tuning
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_selected, y)

        # Best model from grid search
        best_model = grid_search.best_estimator_

        # Cross-validation with the selected features
        cv_scores = cross_val_score(best_model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)

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

        # Print cross-validation results for this number of features
        print(f"Number of Features: {n_features}, CV Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Find the best result based on cross-validated mean accuracy
    best_result = results_df.loc[results_df['cv_mean_accuracy'].idxmax()]
    print(f"\nOptimal Number of Features: {best_result['n_features']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f} ± {best_result['cv_std_accuracy']:.4f}")

    return results_df


def corrupt_dataframe(df):
    """
    Introduce missing values, duplicate rows, non-numerical entries,
    and misspelled strings into a DataFrame with randomly generated fractions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be corrupted.

    Returns:
    - pd.DataFrame: The corrupted DataFrame.
    """
    # Set a random seed for reproducibility
    random.seed(42)
    np.random.seed(42)  # Set seed for numpy-related random operations

    # Introduce missing values
    num_missing = int(0.01 * df.size)
    missing_indices = random.sample(range(df.size), num_missing)

    for idx in missing_indices:
        df.iat[idx // df.shape[1], idx % df.shape[1]] = np.nan

    # Introduce non-numerical entries in numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        num_non_numeric = int(0.0001 * len(df))
        non_numeric_indices = random.sample(range(len(df)), num_non_numeric)

        for idx in non_numeric_indices:
            df.at[idx, col] = random.choice(['a', 'b', 'c', 'invalid', 'non-numeric'])
    
    # Introduce misspellings in string columns
    for col in df.select_dtypes(include='object').columns:
        num_misspellings = int(0.01 * len(df))
        
        # Use random.sample() to select the same indices every time
        misspell_indices = random.sample(range(len(df)), num_misspellings)

        for idx in misspell_indices:
            original_value = df.at[idx, col]
            if isinstance(original_value, str):
                # Randomly generate a misspelled version with the same randomness
                misspelled_value = ''.join(random.choice(original_value) for _ in range(len(original_value)))
                df.at[idx, col] = misspelled_value

    # Introduce duplicate rows
    num_rows_to_duplicate = int(0.05 * len(df))
    if num_rows_to_duplicate > 0:
        duplicates = df.sample(num_rows_to_duplicate, replace=True, random_state=42)  # Set random_state for consistency
        df = pd.concat([df, duplicates], ignore_index=True)

    return df