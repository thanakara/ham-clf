import json

from pathlib import Path

from data.images import image_to_tensor
from ham_clf.model.template import PredictionResponse


def invoke(model, image_path: Path | str) -> str:
    assert image_path.exists()
    jsonpath = Path("src") / "jupyter" / "classes.json"
    with jsonpath.open("r") as f_:
        classes_dict = json.load(f_)
    image = image_to_tensor(image_path=image_path)
    prediction = model.predict(image, verbose=0)
    prediction_class_idx = prediction.argmax(axis=1)[0]
    predicted_class = classes_dict.get(str(prediction_class_idx))
    confidence = float(prediction[0][prediction_class_idx])
    return PredictionResponse(prediction=predicted_class, confidence=confidence)
