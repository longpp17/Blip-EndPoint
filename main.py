import binascii

from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import io
import base64
from PIL import Image
from transformers import pipeline

app = FastAPI()

class ImageType(BaseModel):
    data: List[str] = []

# Load image captioning model on startup
@app.on_event('startup')
async def load_model():
    app.state.captioner = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b-coco")

# Generate API Token Randomly on startup and print that to console
@app.post('/predict')
async def predict(images: ImageType):
    try:
        captions = []
        for image_data in images.data:
            # convert base64 string back into bytes
            try:
                image_bytes = base64.b64decode(image_data)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="Invalid base64 string")

            # open the image
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except OSError:
                raise HTTPException(status_code=400, detail="Invalid image")
            # use the model to predict
            result = app.state.captioner(image)[0]

            # append the caption to the list of captions
            captions.append(result['generated_text'])

        return {'captions': captions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# switching models
@app.post('/switch')
async def switch_model(model_name: str):
    try:
        app.state.captioner = pipeline("image-to-text", model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'model': model_name}

# get current model name
@app.get('/model')
async def get_model():
    return {'model': app.state.captioner.model.name_or_path}

