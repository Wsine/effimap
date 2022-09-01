import numpy as np
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import xgboost as xgb

from arguments import parser
from metric import post_predict, predicates
from model import load_model
from dataset import load_dataset
from fuzz_mutants import SilentConv, RevertBatchNorm, RevertReLu
from utils import get_device, load_pickle_object, load_torch_object, save_object


@torch.no_grad()
def extract_mutant_predicates(ctx, model, dataloader, device):
    model_preds_list = []

    pred_predicates = []
    for mutant_idx in tqdm(range(ctx.num_model_mutants)):
        mutant_path = f'model_mutants/effimap_mutant.{mutant_idx}.pt'
        mutant = load_torch_object(ctx, mutant_path)
        assert(mutant is not None)
        mutant = mutant.to(device)
        mutant.eval()

        predicates_list = []
        for batch_idx, (inputs, targets) in enumerate(
                tqdm(dataloader, desc='Eval', leave=False)):
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

    return pred_predicates


feature_container = []
def general_feature_hook(module, inputs, outputs):
    batch_size = outputs.size(0)
    start_dim = 2 if outputs.dim() == 4 else 1
    dims = [i for i in range(start_dim, outputs.dim())]
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU, nn.Linear)):
        mean = torch.mean(outputs, dim=dims).view(batch_size, -1)
        feature_container.append(mean)
        var = torch.var(outputs, dim=dims).view(batch_size, -1)
        feature_container.append(var)
    if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
        act_ratio = (outputs > 0).sum(dim=dims) / outputs[0].numel()
        act_ratio = act_ratio.view(batch_size, -1)
        feature_container.append(act_ratio)
        in_mask = (inputs[0] < 0).sum(dim=dims) / outputs[0].numel()
        out_mask = (outputs < 0).sum(dim=dims) / outputs[0].numel()
        act_change_ratio = (in_mask - out_mask).view(batch_size, -1)
        feature_container.append(act_change_ratio)


def task_feature_hook(module, inputs, outputs):
    probs = F.softmax(outputs, dim=1)
    # gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
    # feature_container.append(gini)
    # entropy = torch.log(F.softmax(outputs, dim=1).max(dim=1).values).mul(-1.)
    entropy = probs.mul(torch.log(probs)).mul(-1.).sum(dim=1)  # shannon entropy
    entropy = entropy.view(outputs.size(0), -1)
    feature_container.append(entropy)


@torch.no_grad()
def extract_effimap_sample_features(ctx, model, trainloader, device):
    handlers = []
    last_linear = None
    for module in model.modules():
        handler = module.register_forward_hook(general_feature_hook)
        handlers.append(handler)
        if ctx == 'clf' and isinstance(module, nn.Linear):
            last_linear = module
    if ctx == 'clf':
        assert(last_linear is not None)
        handler = last_linear.register_forward_hook(task_feature_hook)
        handlers.append(handler)

    sample_features = []
    for inputs, _ in tqdm(trainloader):
        inputs = inputs.to(device)

        feature_container.clear()
        model(inputs)
        batch_features = torch.cat(feature_container, dim=1)
        sample_features.append(batch_features)
    sample_features = torch.cat(sample_features)

    for h in handlers:
        h.remove()

    sample_features = sample_features.cpu().numpy()
    return sample_features


def train_ranker(ctx, sample_features, mutant_predicates):
    print('features shape:', sample_features.shape)
    print('targets shape:', mutant_predicates.shape)

    assert(len(sample_features) == len(mutant_predicates))
    shuffled = np.random.permutation(len(sample_features))
    sample_features = sample_features[shuffled]
    mutant_predicates = mutant_predicates[shuffled]

    if ctx.task == 'clf':
        ranker = xgb.XGBClassifier(
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='mae',
            tree_method='gpu_hist',
            gpu_id=ctx.gpu
        )
    else:
        raise NotImplemented

    ranker.fit(
        sample_features, mutant_predicates,
        eval_set=[(sample_features, mutant_predicates)]
    )
    save_object(ctx, ranker, 'effimap_ranker.pkl')

    return ranker


def main():
    ctx = parser.parse_args()
    print(ctx)

    device = get_device(ctx)
    model = load_model(ctx, pretrained=True)
    model = model.to(device).eval()

    trainset = load_dataset(ctx, split='train')
    pgd_adv_samples = load_torch_object(ctx, 'pgd_adversarial_samples.pt')
    assert(pgd_adv_samples is not None)
    adv_images, adv_labels = pgd_adv_samples
    advset = torch.utils.data.TensorDataset(adv_images, adv_labels)
    catset = torch.utils.data.ConcatDataset([trainset, advset])
    trainloader = torch.utils.data.DataLoader(
        catset, batch_size=ctx.batch_size, shuffle=False, num_workers=8)

    sample_features = load_pickle_object(ctx, 'effimap_features.pkl')
    if sample_features is None:
        sample_features = extract_effimap_sample_features(ctx, model, trainloader, device)
        save_object(ctx, sample_features, 'effimap_features.pkl')

    mutant_predicates = load_pickle_object(ctx, 'effimap_predicates.pkl')
    if mutant_predicates is None:
        mutant_predicates = extract_mutant_predicates(ctx, model, trainloader, device)
        save_object(ctx, mutant_predicates, 'effimap_predicates.pkl')

    train_ranker(ctx, sample_features, mutant_predicates)


if __name__ == '__main__':
    main()
