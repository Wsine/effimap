import importlib


def load_model(ctx, model_name=None, pretrained=False):
    if model_name is None:
        model_name = ctx.model
    model_module = importlib.import_module(f'models.{ctx.dataset}.{model_name}')
    model = model_module.get_model(ctx, pretrained)
    return model

