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
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

    return (
        accuracy_score,
        nn,
        pd,
        plt,
        precision_score,
        recall_score,
        torch,
        train_test_split,
    )


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
    return X_test, X_train, y_test, y_train


@app.cell
def _(nn):
    class PerceptronModel0(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=15, out_features=5)
            self.layer_2 = nn.Linear(in_features=5, out_features=1)
            self.LReLU = nn.LeakyReLU()

        def forward(self, x):
            return self.layer_2(self.LReLU(self.layer_1(x)))

    return (PerceptronModel0,)


@app.cell
def _(PerceptronModel0, device):
    perceptron_model_0v0 = PerceptronModel0().to(device)
    return (perceptron_model_0v0,)


@app.cell
def _(device, nn, perceptron_model_0v0, torch):
    _weight = 408/502
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([_weight]))
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(perceptron_model_0v0.parameters(), lr=0.001)
    return loss_fn, optimizer


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    loss_fn,
    optimizer,
    perceptron_model_0v0,
    recall_score,
    torch,
    y_test,
    y_train,
):
    #torch.manual_seed(42)
    #TODO: use diff round func for _pred (medical models)
    epochs = 0

    for epoch in range(epochs):
        perceptron_model_0v0.train()

        optimizer.zero_grad()

        _logits = perceptron_model_0v0(X_train).squeeze()
        _loss = loss_fn(_logits, y_train)

        _loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            perceptron_model_0v0.eval()
            with torch.inference_mode():
                _test_logits = perceptron_model_0v0(X_test).squeeze()
                _pred = torch.round(torch.sigmoid(_test_logits))

                print("Recall: ", recall_score(y_test.cpu(), _pred.cpu()))
                print("Accuracy: ", accuracy_score(y_test.cpu(), _pred.cpu()))
                print("====================\n")
    return


@app.cell
def _(torch):
    def threshold_predict(logits, threshold=0.35):  # tunable!
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

    return (threshold_predict,)


@app.cell
def _(
    X_test,
    accuracy_score,
    perceptron_model_0v0,
    precision_score,
    recall_score,
    threshold_predict,
    torch,
    y_test,
):
    perceptron_model_0v0.eval()
    with torch.inference_mode():
        _test_logits = perceptron_model_0v0(X_test).squeeze()
    
        #_pred = torch.round(torch.sigmoid(_test_logits))
        _pred = threshold_predict(logits = _test_logits, threshold=0.18)

        print("Recall: ", recall_score(y_test.cpu(), _pred.cpu()))
        print("Precision: ", precision_score(y_test.cpu(), _pred.cpu()))
        print("Accuracy: ", accuracy_score(y_test.cpu(), _pred.cpu()))
        print("====================\n")
    return


@app.cell
def _(perceptron_model_0v0, torch):
    torch.save(perceptron_model_0v0.state_dict(), 'trained/trained_perceptron_v0.pth')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
