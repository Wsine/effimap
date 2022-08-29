import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xgboost as xgb

from arguments import parser
from metric import post_predict, prediction_error
from model import load_model
from dataset import load_dataloader
from generate_mutants import generate_random_sample_mutants
from utils import get_device, load_pickle_object, save_object


@torch.no_grad()
def extract_model_errors(ctx, model, valloader, device):
    pred_errors = []
    for inputs, targets in tqdm(valloader, desc='Extract'):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_preds = post_predict(ctx, model(inputs))

        errors = prediction_error(ctx, inputs_preds, targets)
        pred_errors.append(errors)
    pred_errors = torch.cat(pred_errors).numpy()
    return pred_errors


def compute_prima_sample_feature(ctx, inputs_preds):
    if ctx.task == 'clf':
        pred_labels, pred_probs = inputs_preds
        sample_label, mutant_labels = pred_labels[0], pred_labels[1:]
        sample_prob, mutant_probs = pred_probs[0], pred_probs[1:]

        f1a = mutant_labels.ne(sample_label).sum()
        f1b = mutant_labels.unique().size(0) - 1
        _, cnt = mutant_labels.unique(return_counts=True)
        f1c = cnt[cnt.topk(2).indices[-1]] if cnt.size(0) > 1 else cnt[0]
        dist = F.cosine_similarity(
            mutant_probs, sample_prob.repeat(mutant_probs.size(0), 1))
        f2a = dist.mean()
        f2b = [
            torch.logical_and(dist > s, dist < s + 0.1).sum()
            for s in torch.linspace(0, 1, steps=10)
        ]
        f2c = (mutant_probs[:, sample_label] - sample_prob[sample_label]).mean()
        feature = torch.Tensor([f1a, f1b, f1c, f2a, *f2b, f2c])
    else:
        raise NotImplemented
    return feature


@torch.no_grad()
def extract_prima_sample_features(ctx, model, valloader, device):
    sample_features = []

    for inputs, _ in tqdm(valloader, desc='Batch'):
        for sample_with_mutants in generate_random_sample_mutants(ctx, inputs):
            inputs = sample_with_mutants.to(device)
            inputs_preds = post_predict(ctx, model(inputs))
            feature = compute_prima_sample_feature(ctx, inputs_preds)
            feature = feature.cpu()
            sample_features.append(feature)

    sample_features = torch.stack(sample_features).numpy()
    save_object(ctx, sample_features, f'prima_features.pkl')

    return sample_features


def train_ranker(ctx, sample_features, model_errors):
    print('features shape:', sample_features.shape)
    print('targets shape:', model_errors.shape)

    assert(len(sample_features) == len(model_errors))
    shuffled = np.random.permutation(len(sample_features))
    sample_features = sample_features[shuffled]
    model_errors = model_errors[shuffled]

    if ctx.task == 'clf':
        ranker = xgb.XGBClassifier(
            eta=0.05,  # learning rate
            colsample_bytree=0.5,
            max_depth=5,
        )
    else:
        raise NotImplemented

    ranker.fit(sample_features, model_errors)
    save_object(ctx, ranker, 'prima_ranker.pkl')

    return ranker


def main():
    ctx = parser.parse_args()
    print(ctx)

    device = get_device(ctx)
    valloader = load_dataloader(ctx, split='val')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    sample_features = load_pickle_object(ctx, f'prima_features.pkl')
    if sample_features is None:
        sample_features = extract_prima_sample_features(ctx, model, valloader, device)

    model_errors = extract_model_errors(ctx, model, valloader, device)

    train_ranker(ctx, sample_features, model_errors)


if __name__ == '__main__':
    main()
