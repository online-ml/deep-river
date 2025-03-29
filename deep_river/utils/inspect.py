from river.utils.inspect import extract_relevant

from deep_river import base


def isdeepestimator_initialized(model):
    return isinstance(extract_relevant(model), base.DeepEstimatorInitialized)
