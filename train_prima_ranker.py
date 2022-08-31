import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xgboost as xgb

from arguments import parser
from metric import post_predict, prediction_error
from model import load_model
from dataset import load_dataloader
from generate_mutants import InverseActivate, generate_random_sample_mutants
from utils import get_device, load_pickle_object, load_torch_object, save_object


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


@torch.no_grad()
def extract_prima_model_mutants_predictions(ctx, valloader, device):
    all_mutants_preds = []
    for mutant_idx in tqdm(range(ctx.num_model_mutants)):
        mutant_path = f'model_mutants/random_mutant.{mutant_idx}.pt'
        mutant = load_torch_object(ctx, mutant_path)
        assert(mutant is not None)
        mutant = mutant.to(device)
        mutant.eval()

        mutant_preds_list = []
        for inputs, targets in tqdm(valloader, desc='Eval', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            mutant_preds = post_predict(ctx, mutant(inputs))
            mutant_preds_list.append(mutant_preds)

        if ctx.task == 'clf':
            mutant_pred_labels = torch.cat([label for label, _ in mutant_preds_list])
            mutant_pred_probs = torch.cat([prob for _, prob in mutant_preds_list])
            all_mutants_preds.append(
                (mutant_pred_labels.cpu(), mutant_pred_probs.cpu())
            )
        else:
            raise NotImplemented

    if ctx.task == 'clf':
        all_mutants_labels = torch.stack([
            labels for labels, _ in all_mutants_preds
        ], dim=-1)
        print('all_mutants_labels -', all_mutants_labels.size())
        all_mutants_probs = torch.stack([
            probs for _, probs in all_mutants_preds
        ], dim=-1)
        print('all_mutants_probs -', all_mutants_probs.size())
        return all_mutants_labels, all_mutants_probs
    else:
        raise NotImplemented


@torch.no_grad()
def extract_prima_sample_mutants_predictions(ctx, model, valloader, device):
    all_mutants_preds = []
    for inputs, _ in tqdm(valloader, desc='Batch'):
        for sample_with_mutants in generate_random_sample_mutants(ctx, inputs):
            inputs = sample_with_mutants.to(device)
            inputs_preds = post_predict(ctx, model(inputs))
            all_mutants_preds.append(inputs_preds)

    if ctx.task == 'clf':
        all_mutants_labels = torch.stack([
            labels for labels, _ in all_mutants_preds
        ]).cpu()
        print('all_mutants_labels -', all_mutants_labels.size())
        all_mutants_probs = torch.stack([
            probs for _, probs in all_mutants_preds
        ]).cpu()
        print('all_mutants_probs -', all_mutants_probs.size())
        return all_mutants_labels, all_mutants_probs
    else:
        raise NotImplemented


def compute_prima_samples_features(ctx, input_pred, samples_pred, models_pred):
    if ctx.task == 'clf':
        input_label, input_probs = input_pred
        samples_labels, sample_probs = samples_pred
        models_labels, models_probs = models_pred

        f1a = [
            models_labels.ne(input_label).sum(),
            samples_labels.ne(input_label).sum()
        ]
        f1b = [
            models_labels.unique().size(0) - 1,
            samples_labels.unique().size(0) - 1
        ]
        _, mm_cnt = models_labels.unique(return_counts=True)
        _, sm_cnt = samples_labels.unique(return_counts=True)
        f1c = [
            mm_cnt[mm_cnt.topk(2).indices[-1]] if mm_cnt.size(0) > 1 else mm_cnt[0],
            sm_cnt[sm_cnt.topk(2).indices[-1]] if sm_cnt.size(0) > 1 else sm_cnt[0]
        ]
        dist_func = lambda a, b: F.cosine_similarity(a, b.repeat(a.size(0), 1))
        mm_dist = dist_func(models_probs, input_probs)
        sm_dist = dist_func(sample_probs, input_probs)
        f2a = [mm_dist.mean(), sm_dist.mean()]
        iterval_func = lambda d, s: torch.logical_and(d > s, d < s + 0.1).sum()
        mm_interval = [iterval_func(mm_dist, s) for s in torch.linspace(0, 1, steps=10)]
        sm_interval = [iterval_func(sm_dist, s) for s in torch.linspace(0, 1, steps=10)]
        f2b = [*mm_interval, *sm_interval]
        f2c_func = lambda a, b: (a - b).abs().mean()
        f2c = [
            f2c_func(models_probs[input_label], input_probs[input_label]),
            f2c_func(sample_probs[input_label], input_probs[input_label]),
        ]
        feature = torch.Tensor([*f1a, *f1b, *f1c, *f2a, *f2b, *f2c])
        assert(feature.size(0) == 30)
    else:
        raise NotImplemented
    return feature


@torch.no_grad()
def extract_prima_sample_features(ctx, model, valloader, device):
    model_mutants_preds = load_torch_object(ctx, 'prima_model_mutants_preds.pt')
    if not model_mutants_preds:
        model_mutants_preds = \
            extract_prima_model_mutants_predictions(ctx, valloader, device)
        save_object(ctx, model_mutants_preds, 'prima_model_mutants_preds.pt')

    sample_mutants_preds = load_torch_object(ctx, 'prima_sample_mutants_preds.pt')
    if not sample_mutants_preds:
        sample_mutants_preds = \
            extract_prima_sample_mutants_predictions(ctx, model, valloader, device)
        save_object(ctx, sample_mutants_preds, 'prima_sample_mutants_preds.pt')

    sample_features = []
    if ctx.task == 'clf':
        model_mutants_labels, model_mutants_probs = model_mutants_preds
        sample_mutants_labels, sample_mutants_probs = sample_mutants_preds
        for mm_labels, mm_probs, sm_labels, sm_probs in zip(
                model_mutants_labels, model_mutants_probs,
                sample_mutants_labels, sample_mutants_probs):
            input_pred = (sm_labels[0], sm_probs[0])
            samples_pred = (sm_labels[1:], sm_probs[1:])
            models_pred = (mm_labels, mm_probs)
            feature = compute_prima_samples_features(
                ctx, input_pred, samples_pred, models_pred)
            sample_features.append(feature)
    else:
        raise NotImplemented

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
