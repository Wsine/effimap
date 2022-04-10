import torch
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from model import get_device
from models.vanilla_vae import VanillaVAE
from utils import *


dispatcher = AttrDispatcher('target')


@dispatcher.register('estimator')
def train_estimator_model(opt):
    X, Y = [], []
    extract_features = load_object(opt, 'extract_features.pt')
    for v in extract_features.values():  # type: ignore
        X.append(v['features'].numpy())
        Y.append(v['mutation'].numpy())

    # hacked to avoid data validation
    X.append(np.zeros_like(X[0]))
    Y.append(np.ones_like(Y[0]))
    X.append(np.ones_like(X[0]))
    Y.append(np.zeros_like(Y[0]))

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    #  for i in range(Y.shape[1]):
    #      y = Y[:, i]
    #      classes_ = np.unique(np.asarray(y))
    #      n_classes_ = len(classes_)
    #      if not np.array_equal(classes_, np.arange(n_classes_)):
    #          print('Fuck', i, classes_, y)
    print('[info] data loaded.')

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

    return multilabel_model, 'mutation_estimator.pkl'


@dispatcher.register('ranking')
def train_ranking_model(opt):
    Y, Z = [], []
    extract_features = load_object(opt, 'extract_features.pt')
    for v in extract_features.values():  # type: ignore
        Y.append(v['mutation'].numpy())
        Z.append(v['prediction'].numpy())
    Y = np.concatenate(Y)
    Z = np.concatenate(Z)
    print(Y.shape, Z.shape)
    print('[info] data loaded.')

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

    return xgb_ranking, 'ranking_model.pkl'


@dispatcher.register('encoder')
def train_autoencoder_model(opt):
    device = get_device(opt)
    trainloader = load_dataloader(opt, split='train')
    img_size = next(iter(trainloader))[0].size(-1)
    model = VanillaVAE(3, img_size, 10).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00005,
        weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.95
    )

    best_rec_err = 1e8
    for e in range(opt.epochs):
        print('Epoch: {}'.format(e))

        loss, rec_err = 0, 0
        for mode in ('Train', 'Validate'):
            model.train() if model == 'Train' else model.eval()
            loss=0; rec_err=0
            tepoch = tqdm(trainloader, desc=mode)
            for batch_idx, (inputs, _) in enumerate(tepoch):
                inputs = inputs.to(device)
                if mode == 'Train':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    tloss = model.loss_function(*outputs, M_N=0.00025)
                    tloss['loss'].backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                    tloss = model.loss_function(*outputs, M_N=1.0)

                loss += tloss['loss'].item()
                rec_err += tloss['Reconstruction_Loss'].item()
                avg_loss = loss / (batch_idx + 1)
                avg_rec_err = rec_err / (batch_idx + 1)
                tepoch.set_postfix(loss=avg_loss, err=avg_rec_err)

        if rec_err < best_rec_err:
            print('Saving...')
            state = {
                'epoch': e,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'err': rec_err
            }
            torch.save(state, get_output_location(opt, 'encoder_model.pt'))
            best_rec_err = rec_err

        scheduler.step()
    return None, ''


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    model, model_name = dispatcher(opt)
    save_object(opt, model, model_name)


if __name__ == '__main__':
    main()
