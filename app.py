from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import gradio as gr

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# load model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# create classifier function
def classify_sentiments(text):
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)

  # Print labels and scores
  probs = {}
  ranking = np.argsort(scores)
  ranking = ranking[::-1]

  for i in range(len(scores)):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    probs[l] = np.round(float(s), 4)
  return probs


#build the Gradio app
Instructuction = "Write an imaginary review about a product or service you might be interested in."
title="Text Sentiment analysis"
description = """Write a Good or Bad reviews of a products or services,\
   the machine learning model is able to predict the text sentiment"""
article = """
            - Click submit button to test sentiment analysis prediction
            - Click clear button to refresh text
           """

gr.Interface(classify_sentiments,
            'text',
            'label',
            title = title,
            description = description,
            Instruction = Instructuction,
            article = article,
            allow_flagging = "never",
            live = False,
            examples=["This has to be the best Introductory course to machine learning",
            "I consider this training an absolute waste of time."]
             ).launch()

