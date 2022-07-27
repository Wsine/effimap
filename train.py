import torch
import numpy as np
import torchvision
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from utils import *


dispatcher = AttrDispatcher('target')


@dispatcher.register('multilabels')
def train_multilabel_model(opt):
    device = get_device(opt)
    model = load_model(opt).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    dataloader = {
        'train': load_dataloader(opt, split='train'),
        'val': load_dataloader(opt, split='val')
    }

    best_acc = 0
    for epoch in range(opt.epochs):
        print('Epoch {}/{}'.format(epoch+1, opt.epochs))
        for phase in ('train', 'val'):
            model.train() if phase == 'train' else model.eval()

            running_loss, running_acc, total = 0, 0, 0
            with tqdm(dataloader[phase], desc=phase) as tbar:
                for batch_idx, (inputs, targets) in enumerate(tbar):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = outputs.detach()
                        loss = criterion(outputs, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_acc += preds.gt(0.5).eq(targets).sum(1).div(targets.size(1)).sum().item()
                    total += targets.size(0)
                    avg_acc = running_acc / total
                    avg_loss = running_loss / (batch_idx + 1)
                    tbar.set_postfix(acc=avg_acc, loss=avg_loss)

            avg_acc = running_acc / total
            if phase == 'val' and avg_acc > best_acc:
                print('Saving model...')
                best_acc = avg_acc
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'acc': avg_acc
                }
                save_object(opt, state, 'multilabels_model.pt')
        scheduler.step()

    return None, ''


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

    if opt.task == 'regress':
        xgb_estimator = xgb.XGBRegressor(
            n_estimators=1000,
            tree_method='gpu_hist',
            gpu_id=opt.gpu,
            verbosity=2
        )
        print(xgb_estimator.get_params())
        multilabel_model = MultiOutputRegressor(xgb_estimator, n_jobs=8)
        multilabel_model.fit(X, Y)
        print('[info] model trained.')

        acc = mean_absolute_error(Y, multilabel_model.predict(X))
        print('MSE on training data: {:.8f}'.format(acc))
    else:
        xgb_estimator = xgb.XGBClassifier(
            use_label_encoder=False,
            n_estimators=1000,
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
    valloader = load_dataloader(opt, split='val', shuffle=True)
    sample_img = next(iter(valloader))[0][0]
    img_channels, img_size = sample_img.size(0), sample_img.size(-1)
    if img_size < 32:
        pad = torchvision.transforms.Pad((32 - img_size) // 2)
        img_size = 32
    else:
        pad = None
    encoder = VanillaVAE(img_channels, img_size, 10).to(device)
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
                if pad is not None:
                    inputs = pad(inputs)
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
                        # torch.normal(0, 0.8, inputs[0].size()).to(device)
                        for e in equals
                        # blur(outputs[0][i].detach()) - outputs[0][i]
                        # for i, e in enumerate(equals)
                    ])
                    outputs[0] = outputs[0] + noises
                    tloss = encoder.loss_function(*outputs, M_N=0.00025)
                    # ploss = torch.logical_xor(
                    #     model(outputs[0]).max(1).indices.eq(predicted), equals).sum()
                    tloss['loss'].backward()
                    # (ploss + tloss['loss']).backward()
                    optimizer.step()
                    # loss += (tloss['loss'].item() + ploss.item())
                else:
                    with torch.no_grad():
                        outputs = encoder(inputs)
                        indices = equals.nonzero(as_tuple=True)
                        for i, o in enumerate(outputs):
                            outputs[i] = o[indices]
                        tloss = encoder.loss_function(*outputs, M_N=1.0)
                    # loss += tloss['loss'].item()

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

    if opt.task == 'regress':
        xgb_ranking = xgb.XGBRegressor(
            eta=0.05,  # learning rate
            colsample_bytree=0.5,
            max_depth=5,
            gpu_id=opt.gpu,
            verbosity=2
        )
    else:
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

    if opt.task == 'regress':
        acc = mean_absolute_error(Y, xgb_ranking.predict(X))
        print('MSE on training data: {:.8f}'.format(acc))
    else:
        acc = accuracy_score(Y, xgb_ranking.predict(X))
        print('Accuracy on training data: {:.4f}%'.format(acc * 100))

    return xgb_ranking, 'prima_ranking_model.pkl'


@dispatcher.register('finetune')
def finetune_model(opt):
    guard_folder(opt)
    device = get_device(opt)
    model = load_model(opt).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if 'tinyimagenet' in opt.dataset:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.005, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    dataloader = {
        'train': load_dataloader(opt, split='train'),
        'val': load_dataloader(opt, split='val')
    }

    num_epochs = 7 if 'tinyimagenet' in opt.dataset else 100
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ('train', 'val'):
            model.train() if phase == 'train' else model.eval()

            running_loss, correct, total = 0, 0, 0
            with tqdm(dataloader[phase], desc=phase) as tbar:
                for batch_idx, (inputs, targets) in enumerate(tbar):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = outputs.max(1)
                        loss = criterion(outputs, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    correct += preds.eq(targets).sum().item()
                    total += targets.size(0)
                    acc = 100. * correct / total
                    avg_loss = running_loss / (batch_idx + 1)
                    tbar.set_postfix(acc=acc, loss=avg_loss)

            acc = 100. * correct / total
            if phase == 'val' and acc > best_acc:
                print('Saving model...')
                best_acc = acc
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'acc': acc
                }
                save_object(opt, state, 'finetune_model.pt')
        scheduler.step()

    return None, ''


@dispatcher.register('regressor')
def convert_model_as_regressor(opt):
    guard_folder(opt)
    device = get_device(opt)
    model = load_model(opt).to(device)


    last_fc_name = [n for n, m in model.named_modules() \
                if isinstance(m, torch.nn.Linear)][-1]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        rgetattr(model, last_fc_name).parameters(), lr=0.001, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    dataloader = {
        'train': load_dataloader(opt, split='train'),
        'val': load_dataloader(opt, split='val')
    }

    num_epochs = 30
    best_loss = 1e8
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ('train', 'val'):
            model.train() if phase == 'train' else model.eval()

            running_loss, total = 0, 0
            with tqdm(dataloader[phase], desc=phase) as tbar:
                for inputs, targets in tbar:
                    inputs = inputs.to(device)
                    targets = targets.float().to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).view(-1)
                        loss = criterion(outputs, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    total += targets.size(0)
                    avg_loss = running_loss / total
                    tbar.set_postfix(loss=avg_loss)

            avg_loss = running_loss / total
            if phase == 'val' and avg_loss < best_loss:
                print('Saving model...')
                best_loss = avg_loss
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }
                save_object(opt, state, 'regressor_model.pt')
        scheduler.step()

    return None, ''


@dispatcher.register('dissector')
def train_dissector_model(opt):
    device = get_device(opt)
    model = load_model(opt).to(device)
    dataloader = {
        'train': load_dataloader(opt, split='train'),
        'val': load_dataloader(opt, split='val')
    }
    model.eval()

    hook_vec = {}
    def _hook_on_layer(lname):
        def __hook(module, inputs, outputs):
            hook_vec[lname] = outputs.detach().flatten(start_dim=1)
        return __hook

    hook_snapshotor = {}
    if opt.dataset == 'cifar100' and opt.model == 'resnet32':
        hook_layers = ['relu', 'layer1', 'layer2', 'layer3']
    elif opt.dataset == 'tinyimagenet' and opt.model == 'resnet18':
        hook_layers = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError('Not supported combinations now')
    for lname in hook_layers:
        module = rgetattr(model, lname)
        module.register_forward_hook(_hook_on_layer(lname))
    batch_inputs, _ = next(iter(dataloader['val']))
    model(batch_inputs.to(device))
    for lname in hook_layers:
        in_features = hook_vec[lname].size(1)
        out_features = opt.num_classes
        snapshotor = torch.nn.Linear(in_features, out_features).to(device)
        optimizer = torch.optim.SGD(snapshotor.parameters(), lr=0.001, momentum=0.9)
        hook_snapshotor[lname] = {
            'model': snapshotor,
            'optim': optimizer,
            'best_acc': 0,
            'running_loss': 0,
            'correct': 0
        }
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ('train', 'val'):
            for k in hook_snapshotor.keys():
                if phase == 'train':
                    hook_snapshotor[k]['model'].train()
                else:
                    hook_snapshotor[k]['model'].eval()
                hook_snapshotor[k]['running_loss'] = 0
                hook_snapshotor[k]['correct'] = 0

            total = 0
            for inputs, targets in tqdm(dataloader[phase], desc=phase):
                inputs, targets = inputs.to(device), targets.to(device)
                for k in hook_snapshotor.keys():
                    hook_snapshotor[k]['optim'].zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    model(inputs)
                    for k in hook_snapshotor.keys():
                        sinputs = hook_vec[k]
                        outputs = hook_snapshotor[k]['model'](sinputs)
                        _, preds = outputs.max(1)
                        loss = criterion(outputs, targets)
                        if phase == 'train':
                            loss.backward()
                            hook_snapshotor[k]['optim'].step()
                        hook_snapshotor[k]['running_loss'] += loss.item()
                        hook_snapshotor[k]['correct'] += \
                                preds.eq(targets).sum().item()
                total += targets.size(0)

            if phase == 'val':
                for k in hook_snapshotor.keys():
                    acc = 100. * hook_snapshotor[k]['correct'] / total
                    if acc > hook_snapshotor[k]['best_acc']:
                        print(f'Updated snapshotor {k} with acc: {acc:.4f}%...')
                        hook_snapshotor[k]['best_acc'] = acc
    state = {
        k: {'net': hook_snapshotor[k]['model'].state_dict()}
        for k in hook_snapshotor.keys()
    }
    return state, 'snapshotors.pt'


@dispatcher.register('transfer')
def transfer_model_to_sub_class(opt):
    guard_folder(opt)
    device = get_device(opt)
    model = load_model(opt).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=0.001, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dataloader = {
        'train': load_dataloader(opt, split='train'),
        'val': load_dataloader(opt, split='val')
    }

    num_epochs = 7
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ('train', 'val'):
            model.train() if phase == 'train' else model.eval()

            running_loss, correct, total = 0, 0, 0
            with tqdm(dataloader[phase], desc=phase) as tbar:
                for batch_idx, (inputs, targets) in enumerate(tbar):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = outputs.max(1)
                        loss = criterion(outputs, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    correct += preds.eq(targets).sum().item()
                    total += targets.size(0)
                    acc = 100. * correct / total
                    avg_loss = running_loss / (batch_idx + 1)
                    tbar.set_postfix(acc=acc, loss=avg_loss)

            acc = 100. * correct / total
            if phase == 'val' and acc > best_acc:
                print('Saving model...')
                best_acc = acc
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'acc': acc
                }
                save_object(opt, state, 'finetune_model.pt')
        scheduler.step()

    return None, ''


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)
    guard_folder(opt)

    model, model_name = dispatcher(opt)
    save_object(opt, model, model_name)


if __name__ == '__main__':
    main()
