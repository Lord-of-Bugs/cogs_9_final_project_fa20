import numpy as np

def assign_severity_level_score(df, column):
    """This functions assigns scores according to the level of severity of the victim's injury.
    
    Parameters:
    ----------
    df: pandas.DataFrame()
        a dataframe that the data is to be drawn from
    column: str
        the name of the column that contains the degrees of severity
        
    Returns:
    -------
    degree_of_severity_score: np.ndarray
        an array containing numerical values from 0 to 4 that classifies
        the degree of severity of the victim.
    """
    
    # Pre-define an empty array for storing severity scores.
    degree_of_severity_score = np.array([])
    
    # Loops through the given column, assign different scores to the 5 different extents of severity. 
    for i in df.get(column):
        if i == 'no injury':
            i = 0
            degree_of_severity_score = np.append(degree_of_severity_score, i)
        elif i == 'complaint of pain':
            i = 1
            degree_of_severity_score = np.append(degree_of_severity_score, i)
        elif i == 'other visible injury':
            i = 2
            degree_of_severity_score = np.append(degree_of_severity_score, i)
        elif i == 'severe injury':
            i = 3
            degree_of_severity_score = np.append(degree_of_severity_score, i)
        elif i == 'killed':
            i = 4
            degree_of_severity_score = np.append(degree_of_severity_score, i)
        else:
            raise ValueError('Undefined value.')
            
    return degree_of_severity_score


def sum_weighted_score(df, column_1, column_2, column_3):
    """Calculates an overall weighted severity score by the number of 
    deaths and the number of injured. This function should only work and 
    make sense for our specified situation.
    
    Returns:
    -------
    an array containing the sum of killed_score and injured_score.
    """
        
    # Pre-define empty arrays for storing weighted scores.
    killed_score = np.array([])
    injured_score = np.array([])
    
    # Scale the square of each death by 4 points (most severe).
    for i in df.get(column_1):
        killed_score = np.append(killed_score, (i ** 2) * 4)

            
    # Multiply party_number_injured with the actual severity_score.       
    for j in zip(df.get(column_2), df.get(column_3)):
        injured_score = np.append(injured_score, j[0] * j[1])
    
    # Return the sum of two sets of scores
    return killed_score + injured_score


def independent_samples_t(cleaned_data_1, cleaned_data_2):
    """This is a function written for performing an independent-samples t-test.
    
    Parameters:
    ----------
    cleaned_data_1: list or np.ndarray
        a list/array that contains only cleaned, numerical values.
    cleaned_data_2: list or np.ndarray
        a list/array that contains only cleaned, numerical values.
        
    Returns:
    -------
    t-statistic: float
        a floating point value that is your calculated t-statistic.
    """
    
    degree_freedom_1 = len(cleaned_data_1) - 1
        
    degree_freedom_2 = len(cleaned_data_2) - 1
        
    sample_mean_1 = np.mean(cleaned_data_1)
        
    sample_mean_2 = np.mean(cleaned_data_2)
        
    sum_of_squares_1 = np.sum((cleaned_data_1 - sample_mean_1)**2)
        
    sum_of_squares_2 = np.sum((cleaned_data_2 - sample_mean_2)**2)
        
    pooled_variance = (sum_of_squares_1 + sum_of_squares_2) / (degree_freedom_1 + degree_freedom_2)
        
    standard_error = np.sqrt((pooled_variance / len(cleaned_data_1)) + (pooled_variance / len(cleaned_data_2)))
        
    try:
            
        t_statistic = (float(sample_mean_1) - float(sample_mean_2)) / float(standard_error)
            
    except ZeroDivisionError:
            
        raise ZeroDivisionError('t-test is insuccessful since the datasets have a standard error of 0.')
        
    return t_statistic