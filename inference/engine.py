
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_fn(test_loader, model, device):
    preds1 = []
    model.eval()
    model.to(device)
    for inputs in test_loader:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds1 = model(inputs)
    preds1.append(torch.max(y_preds1, 1)[1].to('cpu'))
    predictions = []
    for pred in preds1:
        prediction = pred.tolist()
        predictions.extend(prediction)
    return predictions
