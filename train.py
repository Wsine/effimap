import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

from arguments import parser
from utils import *


class AutoFeatureDataset(Dataset):
    def __init__(self, opt):
        self.num_input_mutants = opt.num_input_mutants
        base_folder = get_output_location(opt, 'extract_features')
        self.features = self.load_model_features(base_folder, 0)
        self.groundtruth = self.load_model_mutation(base_folder, 0)
        self.mutation = torch.stack([
            self.load_model_mutation(base_folder, i)
            for i in range(1, opt.num_model_mutants + 1)
        ], dim=1)
        assert(self.features.size(0) == self.mutation.size(0))

    def load_model_features(self, base, idx):
        feature_path = os.path.join(base, f'features_{idx}.pt')
        with open(feature_path, 'rb') as f:
            feature = torch.load(f, map_location='cpu')['features']
        return feature

    def load_model_mutation(self, base, idx):
        feature_path = os.path.join(base, f'features_{idx}.pt')
        with open(feature_path, 'rb') as f:
            mutation = torch.load(f, map_location='cpu')['mutation']
        return mutation

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        feat = self.features[idx].numpy()
        mut  = self.mutation[idx].numpy()
        gt = self.groundtruth[idx].numpy()
        return feat, mut, gt


def train_estimator(opt):
    dataset = AutoFeatureDataset(opt)
    print('[info] data loaded.')

    X = np.stack([x for x, _, _ in dataset])
    Y = np.stack([y for _, y, _ in dataset])

    xgb_estimator = xgb.XGBClassifier(
        use_label_encoder=False,
        objective='binary:logistic',
        eval_metric='logloss',
        #  max_delta_step=5,  # tuning for 0-10 for data unbalanced
        tree_method='gpu_hist',
        gpu_id=opt.gpu,
        verbosity=2
    )
    multilabel_model = MultiOutputClassifier(xgb_estimator, n_jobs=8)
    multilabel_model.fit(X, Y)
    print('[info] model trained.')

    acc = accuracy_score(Y, multilabel_model.predict(X))
    print('Accuracy on training data: {:.4f}%'.format(acc * 100))

    return multilabel_model


def train_ranking_model(opt):
    dataset = AutoFeatureDataset(opt)
    print('[info] data loaded.')

    Y = np.stack([y for _, y, _ in dataset])
    Z = np.stack([z for _, _, z in dataset])

    xgb_ranking = xgb.XGBClassifier(
        use_label_encoder=False,
        eta=0.05,  # learning rate
        colsample_bytree=0.5,
        max_depth=5,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='gpu_hist',
        gpu_id=opt.gpu,
        verbosity=2
    )
    xgb_ranking.fit(Y, Z)
    print('[info] model trained.')

    acc = accuracy_score(Z, xgb_ranking.predict(Y))
    print('Accuracy on training data: {:.4f}%'.format(acc * 100))

    return xgb_ranking


def main():
    opt = parser.parse_args()
    print(opt)

    print('[info] training estimator')
    model = train_estimator(opt)
    save_object(opt, model, 'mutation_estimator.pkl')
    print('[info] training ranking model')
    model = train_ranking_model(opt)
    save_object(opt, model, 'ranking_model.pkl')


if __name__ == '__main__':
    main()
