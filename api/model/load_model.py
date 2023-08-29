import os
from google.cloud import storage
from transformers import AutoModelForSequenceClassification



def load_model():
    if not os.path.exists('local_model'):
        os.makedirs('local_model')
    client = storage.Client()
    bucket = client.get_bucket('vv-2')
    blob = bucket.blob('pytorch_model.bin')
    blob.download_to_filename('local_model/pytorch_model.bin')
    clob = bucket.blob('config.json')
    clob.download_to_filename('local_model/config.json')
    try:
        MODEL_PATH = "local_model"

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return model

    except:
        print(f"\n❌ No model found in GCS bucket")

        return None
