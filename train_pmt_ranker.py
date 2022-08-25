import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from arguments import parser
from metric import post_predict, predicates
from model import load_model
from dataset import load_dataloader
from generate_mutants import InverseActivate
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


def train_ranker(ctx, valloader, pred_predicates):
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
    print('features shape:', sample_features.shape)

    mutation_scores = np.sum(pred_predicates, axis=1) / ctx.num_model_mutants
    print('targets shape:', mutation_scores.shape)

    assert(len(sample_features) == len(mutation_scores))
    shuffled = np.random.permutation(len(sample_features))
    sample_features = sample_features[shuffled]
    mutation_scores = mutation_scores[shuffled]

    regressor = RandomForestRegressor()
    regressor.fit(sample_features, mutation_scores)
    save_object(ctx, regressor, f'pmt_ranker.{ctx.cross_model}.pkl')

    return regressor


def main():
    parser.add_argument('cross_model', type=str, choices=('resnet56', 'vgg13'))
    ctx = parser.parse_args()
    print(ctx)

    valloader = load_dataloader(ctx, split='val')

    pred_predicates = load_pickle_object(ctx, f'pmt_predicates.{ctx.cross_model}.pkl')
    if pred_predicates is None:
        device = get_device(ctx)
        model = load_model(ctx, ctx.cross_model, pretrained=True).to(device)
        model.eval()
        pred_predicates = extract_mutant_predicates(ctx, model, valloader, device)

    train_ranker(ctx, valloader, pred_predicates)


if __name__ == '__main__':
    main()
