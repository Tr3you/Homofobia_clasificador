import pandas as pd


def random_stratified_sampling(df, y, test_size):
    classes = df[y].dropna().unique()
    size = round(round(df.shape[0]*test_size)/len(classes))
    train = df
    test = pd.DataFrame(columns=df.columns)

    for Class in classes:
        pop = train[train[y] == Class]
        sample = pop.sample(n=size)
        test = pd.concat([test, sample])
        train = train.drop(sample.index)

    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)

    x_train = train.drop(axis=1, columns=[y])
    y_train = train[y]
    x_test = test.drop(axis=1, columns=[y])
    y_test = test[y]

    return x_train, y_train, x_test, y_test
