"""Train the original 261127 Roboflow GT-polygon CORAL. Training is opt-in."""

import argparse
from collections import Counter
import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from coral_model import CoralNet, targets_for, predict_coral_from_features
from evaluate_coral import evaluate
from polygon_features import FEATURES, extract_gt_features, pairs_for


SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 250
PATIENCE = 35
LEARNING_RATE = 1e-3
USE_BALANCED_SAMPLER = False


def fit(train_df, valid_df, output, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    checkpoint_path = output / "coral_gt_best.pt"
    DEVICE = device
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(train_df[FEATURES].to_numpy(np.float32)).astype(np.float32)
    xvalid=scaler.transform(valid_df[FEATURES].to_numpy(np.float32)).astype(np.float32)
    ytrain=train_df['hr_grade'].to_numpy(np.int64)
    yvalid=valid_df['hr_grade'].to_numpy(np.int64)

    model=CoralNet(len(FEATURES)).to(DEVICE)
    dataset=TensorDataset(torch.from_numpy(xtrain),torch.from_numpy(ytrain))
    if USE_BALANCED_SAMPLER:
        freq=Counter(ytrain.tolist())
        weights=torch.tensor([1.0/freq[int(y)] for y in ytrain],dtype=torch.double)
        sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
        loader=DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler)
    else:
        loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)
    xv=torch.from_numpy(xvalid).to(DEVICE)
    yv=torch.from_numpy(yvalid).to(DEVICE)
    best_loss=float('inf')
    best_state=None
    wait=0
    history=[]
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        epoch_loss=0.0
        for bx,by in loader:
            bx=bx.to(DEVICE)
            by=by.to(DEVICE)
            optimizer.zero_grad()
            logits=model(bx)
            loss=F.binary_cross_entropy_with_logits(logits,targets_for(by))
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()*len(by)
        model.eval()
        with torch.no_grad():
            val_loss=F.binary_cross_entropy_with_logits(model(xv),targets_for(yv)).item()
        history.append({'epoch':epoch,'train_loss':epoch_loss/len(dataset),'valid_loss':val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss=val_loss
            best_state=copy.deepcopy(model.state_dict())
            best_epoch=epoch
            wait=0
        else:
            wait+=1
        if epoch==1 or epoch%10==0:
            print(f'Epoch {epoch:3d} | Train BCE {epoch_loss/len(dataset):.4f} | Valid BCE {val_loss:.4f}')
        if wait>=PATIENCE:
            print(f'Early stopping: epoch {epoch} | best {best_epoch}')
            break

    model.load_state_dict(best_state)
    torch.save({
        'model_state':best_state,
        'features':FEATURES,
        'scaler_mean':scaler.mean_.tolist(),
        'scaler_scale':scaler.scale_.tolist(),
        'best_epoch':best_epoch,
        'best_valid_bce':best_loss
    },checkpoint_path)
    pd.DataFrame(history).to_csv(output/'coral_training_history.csv',index=False)
    print('CORAL 저장:', checkpoint_path)
    model.eval()
    return model, scaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    pairs = {split: pairs_for(a.dataset, split) for split in ("train", "valid", "test")}
    ids = {split: {image.stem.casefold() for image, _ in items} for split, items in pairs.items()}
    if any(ids[x] & ids[y] for x, y in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        p.error("An image stem occurs in multiple splits")
    if a.output.resolve().is_relative_to(a.dataset.resolve()):
        p.error("Write to a new directory outside the input dataset")
    a.output.mkdir(parents=True)
    frames = {split: extract_gt_features(split, pairs[split]) for split in ("train", "valid")}
    if frames["train"].hr_grade.nunique() < 2:
        p.error("Training requires the original HR0–HR4 polygon labels")
    for split, frame in frames.items():
        frame.to_csv(a.output / f"features_{split}_gt.csv", index=False, encoding="utf-8-sig")
    if a.train:
        device = torch.device(a.device)
        model, scaler = fit(frames["train"], frames["valid"], a.output, device)
        result = predict_coral_from_features(frames["valid"], model, scaler, device)
        result.to_csv(a.output / "valid_GT_polygon_CORAL_predictions.csv", index=False)
        import json
        (a.output / "valid_GT_CORAL_metrics.json").write_text(
            json.dumps(evaluate(result, "gt"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
