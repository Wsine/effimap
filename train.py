import torch
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from model import get_device, load_model
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
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    print('[info] data loaded.')

    xgb_estimator = xgb.XGBClassifier(
        use_label_encoder=False,
        n_estimators=800,
        objective='binary:logistic',
        eval_metric='logloss',
        max_delta_step=5,  # tuning for 0-10 for data unbalanced
        tree_method='gpu_hist',
        gpu_id=opt.gpu,
        verbosity=2
    )
    print(xgb_estimator.get_params())
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
    model = load_model(opt).to(device)
    model.eval()
    valloader = load_dataloader(opt, split='val')
    img_size = next(iter(valloader))[0].size(-1)
    encoder = VanillaVAE(3, img_size, 10).to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
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
            encoder.train() if encoder == 'Train' else encoder.eval()
            loss=0; rec_err=0
            tepoch = tqdm(valloader, desc=mode)
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    _, predicted = model(inputs).max(1)
                equals = predicted.eq(targets)
                if mode == 'Train':
                    optimizer.zero_grad()
                    outputs = encoder(inputs)
                    noises = torch.stack([
                        torch.zeros_like(inputs[0]) if e else \
                        torch.logical_and(
                            torch.bernoulli(
                                torch.zeros_like(inputs[0]).fill_(0.1)),
                            torch.normal(0, 0.8, inputs[0].size()).to(device)
                        )
                        for e in equals
                    ])
                    outputs[0] = outputs[0] + noises
                    tloss = encoder.loss_function(*outputs, M_N=0.00025)
                    tloss['loss'].backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = encoder(inputs)
                        indices = equals.nonzero(as_tuple=True)
                        for i, o in enumerate(outputs):
                            outputs[i] = o[indices]
                        tloss = encoder.loss_function(*outputs, M_N=1.0)

                loss += tloss['loss'].item()
                rec_err += tloss['Reconstruction_Loss'].item()
                avg_loss = loss / (batch_idx + 1)
                avg_rec_err = rec_err / (batch_idx + 1)
                tepoch.set_postfix(loss=avg_loss, err=avg_rec_err)

        if rec_err < best_rec_err:
            print('Saving...')
            state = {
                'epoch': e,
                'net': encoder.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'err': rec_err
            }
            torch.save(state, get_output_location(opt, 'encoder_model.pt'))
            best_rec_err = rec_err

        scheduler.step()
    return None, ''


@dispatcher.register('l2r')
def prima_learning_to_rank(opt):
    input_features = load_object(opt, 'prima_input_features_val.pt')
    model_features = load_object(opt, 'prima_model_features_val.pt')
    feature_target = load_object(opt, 'prima_feature_target_val.pt')
    X = torch.cat(
        (input_features['feats'], model_features['feats']),  # type: ignore
        dim=1).numpy()
    Y = feature_target['equals'].numpy()  # type: ignore
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
    xgb_ranking.fit(X, Y)
    print('[info] model trained.')

    acc = accuracy_score(Y, xgb_ranking.predict(X))
    print('Accuracy on training data: {:.4f}%'.format(acc * 100))

    return xgb_ranking, 'prima_ranking_model.pkl'


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    model, model_name = dispatcher(opt)
    save_object(opt, model, model_name)


if __name__ == '__main__':
    main()
