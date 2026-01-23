import os
import tempfile

from typing import Annotated
from pathlib import Path

import tensorflow as tf

from fastapi import File, FastAPI, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse

from ham_clf.model.endpoint import invoke
from ham_clf.model.template import PredictionResponse

api = FastAPI()

model_path = Path("src") / "jupyter" / "mobilenetv2_ft0.keras"
model = tf.keras.models.load_model(model_path)


@api.get("/", include_in_schema=False)
def home():
    content = """
    <h1>Skin Cancer [HAM10000] Classifier</h1>
    <p>Upload an image to get a prediction</p>
    <form action="/model/invoke" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Predict">
    </form>
    """
    return HTMLResponse(content=content)


@api.post("/model/invoke", response_model=PredictionResponse)
async def invoke_endpoint(file: Annotated[UploadFile, File()]):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an Image")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        response: PredictionResponse = invoke(model=model, image_path=Path(tmp_path))
        os.unlink(tmp_path)
        return response  # auto - no need of JSONResponse(response.model_dump())
    except Exception as e:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e
