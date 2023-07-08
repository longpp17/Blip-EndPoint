from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import io
import base64
from PIL import Image
from transformers import pipeline
import os
app = FastAPI()

class ImageType(BaseModel):
    data: List[str] = []
    token: str

# Load image captioning model on startup
@app.on_event('startup')
async def load_model():
    global captioner
    captioner = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b-coco")

# Generate API Token Randomly on startup and print that to console
@app.on_event('startup')
async def generate_token():
    global token
    token = base64.b64encode(os.urandom(32)).decode('utf-8')
    print(f'API Token: {token}')


@app.post('/predict')
async def predict(images: ImageType):
    try:
        captions = []
        if images.token != token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        else:
            for image_data in images.data:
                # convert base64 string back into bytes
                image_bytes = base64.b64decode(image_data)

                # open the image
                image = Image.open(io.BytesIO(image_bytes))

                # use the model to predict
                result = captioner(image)[0]

                # append the caption to the list of captions
                captions.append(result['generated_text'])

            return {'captions': captions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
