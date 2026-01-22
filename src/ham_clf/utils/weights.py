import numpy as np

from sklearn.utils.class_weight import compute_class_weight


def get_class_weight_dict(classes: list[str], y: np.ndarray) -> dict[str, float]:
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = dict(enumerate(weights))
    return class_weight_dict
