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
        return feat, mut


def train_estimator(opt):
    dataset = AutoFeatureDataset(opt)
    print('[info] data loaded.')

    X = np.stack([x for x, _ in dataset])
    Y = np.stack([y for _, y in dataset])
    xgb_estimator = xgb.XGBClassifier(
        use_label_encoder=False,
        objective='binary:logistic',
        eval_metric='logloss',
        #  max_delta_step=0,  # tuning for 0-10 for data unbalanced
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


def main():
    opt = parser.parse_args()
    print(opt)

    model = train_estimator(opt)
    save_object(opt, model, 'mutation_estimator.pkl')


if __name__ == '__main__':
    main()
