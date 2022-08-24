import importlib


def load_model(ctx, pretrained=False):
    model_module = importlib.import_module(f'models.{ctx.dataset}.{ctx.model}')
    model = model_module.get_model(ctx, pretrained)
    return model

