import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
import pyttsx3
from twilio.rest import Client

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")


def callPolice():
    account_sid = "AC2bd6510d6b2814b36c5e09a896dcdda4"
    auth_token = "7d47f4bf2b25a338d8c8ccf1634a6f2a"
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        twiml="<Response><Say>Hello. Help me I am getting harassed.</Say></Response>",
        to="+919999988888",
        from_="+16205019690"
    )

    print(call.sid)


def detect_abusive_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class == 1


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print("Error:", e)
        return None


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        text = speech_to_text()
        if text is not None:
            if detect_abusive_language(text):
                response = "Harassment Detected!"
                print(response)
                text_to_speech(response)
                callPolice()


if __name__ == "__main__":
    main()
