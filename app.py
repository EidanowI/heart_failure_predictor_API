from fastapi import FastAPI
from typing import Dict, Any
import torch
from torch import nn
import pandas as pd

class PerceptronModel0(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=15, out_features=5)
            self.layer_2 = nn.Linear(in_features=5, out_features=1)
            self.LReLU = nn.LeakyReLU()

        def forward(self, x):
            return self.layer_2(self.LReLU(self.layer_1(x)))

app = FastAPI()
perceptron_model = PerceptronModel0()
perceptron_model.load_state_dict(torch.load('Models/trained/trained_perceptron_v0.pth'))
perceptron_model.eval()

def threshold_predict(logits, threshold=0.35):
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()

@app.post('/perceptron/predict')
def perceptron_prediction(data: Dict[str, Any]):
    df = pd.DataFrame([data])

    template = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
    categ_col = df.select_dtypes(include=['object']).columns.tolist()
    dummed_df = pd.get_dummies(data=df, drop_first=True, dtype=float, columns=categ_col)
    dummed_df = dummed_df.reindex(columns=template, fill_value=0)
    X = torch.from_numpy(dummed_df.to_numpy()).type(torch.float)

    logits = perceptron_model(X).squeeze()
    pred = threshold_predict(logits = logits, threshold=0.18)

    if pred.item() == 0:
        return {"message": "Healthy"}
    else:
        return {"message": "Risk"}