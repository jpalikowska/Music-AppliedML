
import numpy as np

def  train_test_split(df, training_data_fraction, shuffle=True, return_numpy=True):
    '''
    Split all available data into a training and test data set.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of available data.
    training_data_fraction : float
        The fraction of the data that should be used for training.
        E.g., `training_data_fraction=0.6` means 60%.
    shuffle : bool , optional
        If True, randomly reshuffles the data before the split.
    return_numpy : bool, optional
        If True, returns the shuffled dataset and the splits as numpy arrays.
    '''
    
    if shuffle is True:
        df_ = df.sample(frac=1).reset_index(drop=True)
    else:
        df_= df
    
    train_df = df_.iloc[:round(training_data_fraction*len(df_)),:]
    test_df = df_.iloc[round(training_data_fraction*len(df_)):,:]

    if return_numpy:
        X = df_.iloc[:, 1:-1].to_numpy()
        Y = df_['Class'].to_numpy()

        train_X = train_df.iloc[:, 1:-1].to_numpy()
        train_Y = train_df['Class'].to_numpy()

        test_X = test_df.iloc[:, 1:-1].to_numpy()
        test_Y = test_df['Class'].to_numpy()

        return X, Y, train_X, train_Y, test_X, test_Y
    else:
        return df_, train_df, test_df


def k_fold_split(X,Y,k):
    perm_ind = np.random.permutation(np.arange(X.shape[0]))
    return np.array_split(X[perm_ind],k), np.array_split(Y[perm_ind],k)