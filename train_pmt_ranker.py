import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from arguments import parser
from metric import post_predict, predicates
from model import load_model
from dataset import load_dataloader
from generate_mutants import InverseActivate, generate_random_sample_mutants
from utils import get_device, load_pickle_object, load_torch_object, save_object


@torch.no_grad()
def extract_mutant_predicates(ctx, model, valloader, device):
    model_preds_list = []

    pred_predicates = []
    for mutant_idx in tqdm(range(ctx.num_model_mutants)):
        mutant_path = f'model_mutants/random_mutant.{mutant_idx}.pt'
        mutant = load_torch_object(ctx, mutant_path)
        assert(mutant is not None)
        mutant = mutant.to(device)
        mutant.eval()

        predicates_list = []
        for batch_idx, (inputs, targets) in enumerate(
                tqdm(valloader, desc='Eval', leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx > len(model_preds_list) - 1:
                model_preds = post_predict(ctx, model(inputs))
                model_preds_list.append(model_preds)
            else:
                model_preds = model_preds_list[batch_idx]

            mutant_preds = post_predict(ctx, mutant(inputs))
            pdcs = predicates(ctx, model_preds, mutant_preds)
            predicates_list.append(pdcs)
        predicates_list = torch.cat(predicates_list)

        pred_predicates.append(predicates_list)

    pred_predicates = torch.stack(pred_predicates)
    pred_predicates = pred_predicates.transpose(0, 1).numpy()
    save_object(ctx, pred_predicates, f'pmt_predicates.{ctx.cross_model}.pkl')

    return pred_predicates


def extract_pmt_sample_features(ctx, valloader):
    sample_features = []
    for inputs, _ in valloader:
        f1to5 = [ctx.num_model_mutants, 1, ctx.num_model_mutants, 4, 0]
        f6to10 = [1, 1, 0, 10, 0]
        f11to25 = [0, 0, 1, 0, 0]
        sample_feature = np.array(f1to5 + f6to10 + f11to25)

        batch_size = inputs.size(0)
        batch_features = np.repeat(sample_feature[None, ...], batch_size, axis=0)

        sample_features.append(batch_features)
    sample_features = np.concatenate(sample_features)

    return sample_features


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
    save_object(ctx, sample_features, f'prima_features.{ctx.cross_model}.pkl')

    return sample_features


def train_ranker(ctx, sample_features, pred_predicates):
    mutation_scores = np.sum(pred_predicates, axis=1) / ctx.num_model_mutants

    print('features shape:', sample_features.shape)
    print('targets shape:', mutation_scores.shape)

    assert(len(sample_features) == len(mutation_scores))
    shuffled = np.random.permutation(len(sample_features))
    sample_features = sample_features[shuffled]
    mutation_scores = mutation_scores[shuffled]

    regressor = RandomForestRegressor()
    regressor.fit(sample_features, mutation_scores)

    save_name = f'pmt_ranker.{ctx.cross_model}.{ctx.feature_source}.pkl'
    save_object(ctx, regressor, save_name)

    return regressor


def main():
    parser.add_argument('cross_model', type=str, choices=('resnet56', 'vgg13'))
    parser.add_argument('feature_source', type=str, choices=('pmt', 'prima'))
    ctx = parser.parse_args()
    print(ctx)

    device = get_device(ctx)
    valloader = load_dataloader(ctx, split='val')
    model = load_model(ctx, ctx.cross_model, pretrained=True).to(device)
    model.eval()

    pred_predicates = load_pickle_object(ctx, f'pmt_predicates.{ctx.cross_model}.pkl')
    if pred_predicates is None:
        pred_predicates = extract_mutant_predicates(ctx, model, valloader, device)

    if ctx.feature_source == 'pmt':
        sample_features = extract_pmt_sample_features(ctx, valloader)
    else:
        sample_features = load_pickle_object(ctx, f'prima_features.{ctx.cross_model}.pkl')
        if sample_features is None:
            sample_features = extract_prima_sample_features(ctx, model, valloader, device)

    train_ranker(ctx, sample_features, pred_predicates)


if __name__ == '__main__':
    main()
