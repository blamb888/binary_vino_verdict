from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.model.load_model import load_model
import numpy as np
from transformers import AutoTokenizer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the conversion function
def convert_to_2_scale(arr):
    arr_2_scale = []
    for val in arr:
        if val in [0, 1]:
            arr_2_scale.append(0)  # bad
        else:
            arr_2_scale.append(1)  # average
    return np.array(arr_2_scale)

app.state.model = load_model()


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@app.get("/")
def root():
    return dict(greeting="Welcome!")


@app.get("/predict")
def predict(review):

    model = app.state.model
    assert model is not None

    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_classes = np.argmax(logits, axis=1)
    predicted_classes_2_scale = convert_to_2_scale(predicted_classes)  # convert to 2 scale

    verdict = 'good' if predicted_classes_2_scale[0] else 'bad'

    return {'verdict': verdict}
