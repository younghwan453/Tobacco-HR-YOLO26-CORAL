"""The original Roboflow GT CORAL architecture and inference from 261127coral.ipynb."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

from polygon_features import FEATURES


class CoralNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.score=nn.Sequential(
            nn.Linear(n_features,64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64,16), nn.ReLU(), nn.Linear(16,1,bias=False)
        )
        self.bias0=nn.Parameter(torch.tensor(1.2))
        self.raw_gaps=nn.Parameter(torch.full((3,),0.2))

    def forward(self,x):
        first=self.bias0.view(1)
        remaining=self.bias0-torch.cumsum(F.softplus(self.raw_gaps),dim=0)
        thresholds=torch.cat((first,remaining),dim=0)
        return self.score(x)+thresholds.unsqueeze(0)


def targets_for(y):
    return (y.unsqueeze(1) > torch.arange(4,device=y.device)).float()


def predict_coral_from_features(feature_df, model, scaler, device):
    if feature_df.empty:
        return feature_df.copy()
    x=scaler.transform(feature_df[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        logits=model(torch.from_numpy(x).to(device))
        exceed=torch.sigmoid(logits).cpu().numpy()  # shape=(N,4), P(HR>0~3)
    probs=np.column_stack((1-exceed[:,0],exceed[:,0]-exceed[:,1],
                           exceed[:,1]-exceed[:,2],exceed[:,2]-exceed[:,3],exceed[:,3]))
    probs=np.clip(probs,0,1)
    pred=(exceed>=0.5).sum(axis=1).astype(int)
    expected=exceed.sum(axis=1)
    out=feature_df.copy()
    out['hr_pred']=pred
    out['hr_expected']=expected
    for i in range(5):
        out[f'prob_HR{i}']=probs[:,i]
    return out


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES:
        raise ValueError("Checkpoint feature order differs")
    model = CoralNet(len(FEATURES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(FEATURES)
    return model, scaler
