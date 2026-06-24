from deep_river import base


def isdeepestimator_initialized(model):
    while hasattr(model, "_last_step"):
        model = model._last_step
    return isinstance(extract_relevant(model), base.DeepEstimator)


def extract_relevant(model):
    while hasattr(model, "_last_step"):
        model = model._last_step
    return model
