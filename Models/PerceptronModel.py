import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    return pd, plt, torch, train_test_split


@app.cell
def _(torch):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print(device)
    return (device,)


@app.cell
def _(pd):
    df = pd.read_csv("../dataset/heart.csv")
    return (df,)


@app.cell
def _(df):
    print(df.head())
    print(df.shape)
    return


@app.cell
def _(df, plt):
    sex_counts = df["Sex"].value_counts()
    heartDis_counts = df["HeartDisease"].value_counts()

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
    axs[0].set_title('Sex')
    axs[1].pie(heartDis_counts.values, labels=heartDis_counts.index, autopct='%1.1f%%')
    axs[1].set_title('HeartDisease')
    return


@app.cell
def _(df, plt):
    fig2, axs2 = plt.subplots(1, 2, figsize=(15, 7))

    axs2[0].scatter(df['Age'], df['Oldpeak'], c=df['HeartDisease'], cmap='coolwarm', alpha=0.6, s=50)
    axs2[1].scatter(df['Age'], df['ST_Slope'], c=df['HeartDisease'], cmap='coolwarm', alpha=0.6, s=50)
    axs2[0].set_xlabel('Age')
    axs2[0].set_ylabel('Oldpeak (Депрессия ST-сегмента при нагрузке (0=норма, >2=серьёзно))')
    axs2[0].set_title('Age vs Oldpeak')
    axs2[1].set_xlabel('Age')
    axs2[1].set_ylabel('ST_Slope (Наклон ST-сегмента: 0=восходящий, 1=плоский, 2=нисходящий (риск))')
    axs2[1].set_title('Age vs ST_Slope')
    return


@app.cell
def _(df, pd):
    categ_col = df.select_dtypes(include=['object']).columns.tolist()
    dummed_df = pd.get_dummies(data=df, drop_first=True, dtype=float, columns=categ_col)
    return (dummed_df,)


@app.cell
def _(device, dummed_df, torch):
    X = dummed_df.drop(columns="HeartDisease")
    y = dummed_df["HeartDisease"]
    X = torch.from_numpy(X.to_numpy()).type(torch.float).to(device)
    y = torch.from_numpy(y.to_numpy()).type(torch.float).to(device)
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return


if __name__ == "__main__":
    app.run()
