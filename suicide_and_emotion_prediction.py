from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np
from fastapi import FastAPI

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer
print("Loading the tweet classification model...")
model_path = "models/suicide-tweet-classification-model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
print("Model loading complete")


print("Loading the emotion classification model...")
idx_to_emotion = {
    0:  'admiration 😊', 1:  'amusement 😄', 2:  'anger 😠', 3:  'annoyance 😒', 4:  'approval 👍', 5:  'caring 🤗', 
    6:  'confusion 😕', 7:  'curiosity 🤔', 8:  'desire 😍', 9:  'disappointment 😞', 10: 'disapproval 👎', 
    11: 'disgust 🤢', 12: 'embarrassment 😳', 13: 'excitement 😆', 14: 'fear 😨', 15: 'gratitude 🙏', 
    16: 'grief 😢', 17: 'joy 😃', 18: 'love ❤️', 19: 'nervousness 😬', 20: 'optimism 🌞', 21: 'pride 😌', 
    22: 'realization 💡', 23: 'relief 😌', 24: 'remorse 😔', 25: 'sadness 😥', 26: 'surprise 😲', 27: 'neutral 😐'
}
emotion_model_path = "models/emotion-classification-model"
emotion_tokenizer = RobertaTokenizer.from_pretrained(emotion_model_path)
emotion_model = RobertaForSequenceClassification.from_pretrained(emotion_model_path).to(device)
print("Model loading complete")



def postprocess_suicide_prediction_and_confidence(probs):
    mapping = {0: "Suicide", 1: "Normal"}
    pred_label_idx = probs.argmax()
    confidence = probs[pred_label_idx] * 100.0
        
    prediction = mapping[pred_label_idx]
    return prediction, float(round(confidence, 2))

def predict_suicide(input_text):
    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    probs = probs.cpu().detach().numpy()[0]    
    return postprocess_suicide_prediction_and_confidence(probs)

def predict_emotion(input_text):
    inputs = emotion_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = emotion_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    top_probs, top_indices = torch.topk(probs, k=5)

    # Filter the top predictions based on the 60% threshold
    # filtered_predictions = [(idx_to_label[idx.item()], round(prob.item()*100.0, 2)) for idx, prob in zip(top_indices, top_probs)]
    filtered_predictions = [(idx_to_emotion[idx.item()], float(min(round(prob.item()*100.0, 2), 99.5))) for idx, prob in zip(top_indices, top_probs)]

    # Return the filtered predictions
    return filtered_predictions



@app.get('/')
def read_root():
    return {'message': 'Suicide Intevention API'}


@app.post('/predict')
def predict(input_text: str):
    """
    Predicts the Suicide risk, confidence and associated emotions for a particular string.
    
    Args:
        input_text:A string user types in text box
        e.g. "I want to kill myself"

    Returns:
        dict:
        input_text:text given as input
        prediction:Suicide/Normal
        confidence:the confidence of prediction
        top_k_emotions:emotions user might be feeling
    """        
    prediction, confidence = predict_suicide(input_text)
    top_k_emotions = predict_emotion(input_text)
    output_dict={"input_text":input_text, "prediction":prediction, "confidence":confidence, "top_k_emotions":top_k_emotions}
    return output_dict