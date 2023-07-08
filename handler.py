from typing import Dict, List, Any
from PIL import Image
import torch
import base64
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EndpointHandler():
    def __init__(self, path=""):
        # load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)

        self.image_to_text_pipeline = pipeline('image-to-text', model=model, tokenizer=tokenizer)

        image_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
        Return:
            A :obj:`dict`: will be serialized and returned
        """
        # Extract inputs and kwargs from the data
        inputs = data["inputs"]
        parameters = data.pop("parameters", None)

        # Decode base64 image to PIL
        image = Image.open(BytesIO(base64.b64decode(inputs['image'])))
        image = self.transform(image).unsqueeze(0).to(device)

        # Run the model for prediction
        if parameters is not None:
            predictions = self.image_to_text_pipeline(image, **parameters)
        else:
            predictions = self.image_to_text_pipeline(image)

        return predictions
