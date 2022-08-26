import os
import json
import pickle
import functools
import hashlib

import torch
# import pandas as pd


def get_device(ctx):
    if ctx.device == 'cpu':
        return torch.device('cpu')
    elif ctx.device == 'cuda':
        if torch.cuda.is_available() is False:
            return torch.device('cpu')
        return torch.device(f'cuda:{ctx.gpu}')
    else:
        raise ValueError('Invalid device type')


def get_output_path(ctx, filename=None):
    output_folder = os.path.join(ctx.output_dir, ctx.dataset, ctx.model)
    if isinstance(filename, str):
        return os.path.join(output_folder, filename)
    if isinstance(filename, list):
        return os.path.join(output_folder, *filename)
    return output_folder


def guard_folder(ctx, folder=None):
    output_folder = get_output_path(ctx)
    folder_to_create = [output_folder]
    if isinstance(folder, str):
        folder_to_create += [os.path.join(output_folder, folder)]
    elif isinstance(folder, (list, tuple)):
        folder_to_create += [os.path.join(output_folder, f) for f in folder]
    for p in folder_to_create:
        if not os.path.isdir(p):
            os.makedirs(p)


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def cache_object(filename):
    def _decorator(func):
        def __func_wrapper(*args, **kwargs):
            cache_dir = args[0].cache_dir if hasattr(args[0], 'cache_dir') else 'cache'
            prefix = f'{args[0].dataset}_' if hasattr(args[0], 'dataset') else ''
            filepath = os.path.join(cache_dir, prefix + filename)
            try:
                cache = pickle.load(open(filepath, 'rb'))
            except IOError:
                cache = {}
            paramskey = dict_hash(kwargs)
            if paramskey not in cache:
                cache[paramskey] = func(*args, **kwargs)
                if not os.path.isdir(cache_dir):
                    os.makedirs(cache_dir)
                pickle.dump(cache, open(filepath, 'wb'))
            else:
                print('[info] {} cache hit'.format(func.__name__))
            return cache[paramskey]
        return __func_wrapper
    return _decorator


def preview_object(obj):
    def _reduce_object(obj):
        if isinstance(obj, list):
            return [_reduce_object(o) for o in obj[:2]]
        elif isinstance(obj, dict):
            return {k: _reduce_object(v) for k, v in obj.items()}
        else:
            return obj

    print(json.dumps(_reduce_object(obj), indent=2, ensure_ascii=False))


def save_object(ctx, obj, filename, **kwargs):
    if obj is None:
        return
    mode = 'b' if filename.endswith(('.pkl', '.pt', '.pth')) else ''
    dstdir = os.path.join(ctx.output_dir, ctx.dataset, ctx.model)

    filepath = os.path.join(dstdir, filename)
    with open(filepath, f'w{mode}') as f:
        if filename.endswith('.pkl'):
            pickle.dump(obj, f, **kwargs)
        elif filename.endswith('.json'):
            if 'indent' not in kwargs:
                kwargs['indent'] = 2
            if 'ensure_ascii' not in kwargs:
                kwargs['ensure_ascii'] = False
            json.dump(obj, f, **kwargs)
        elif filename.endswith('.csv'):
            obj.to_csv(f, float_format='%.8f')
        elif filename.endswith('.pt'):
            torch.save(obj, f)
        else:
            raise NotImplemented


def check_file_exists(ctx, filename):
    filepath = os.path.join(get_output_path(ctx), filename)
    return os.path.exists(filepath)


def load_pickle_object(ctx, filename, **kwargs):
    filepath = os.path.join(get_output_path(ctx), filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    return obj


def load_json_object(ctx, filename):
    filepath = os.path.join(get_output_path(ctx), filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def load_torch_object(ctx, filename):
    filepath = os.path.join(get_output_path(ctx), filename)
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, map_location='cpu')


# borrow from: https://stackoverflow.com/questions/31174295/
#   getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# copied from: https://stackoverflow.com/questions/36836161/
#   singledispatch-based-on-value-instead-of-type#36837332
class AttrDispatcher(object):
    def __init__(self, attr):
        self.attr = attr
        self.registry = {}

    def __call__(self, *args, **kwargs):
        ctx, *_ = args
        assert hasattr(ctx, self.attr), f"The first argument must has attribute '{self.attr}'"
        func = self.registry[getattr(ctx, self.attr)]
        return func(*args, **kwargs)

    def register(self, key):
        def _decorator(method):
            self.registry[key] = method
            return method
        return _decorator

