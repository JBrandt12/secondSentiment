from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification 
from scipy.special import softmax
import pandas as pd 

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" 
tokenizer = AutoTokenizer.from_pretrained(MODEL) 
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores(example):
    encoded_text = tokenizer(example, return_tensors='pt') 
    output = model(**encoded_text) 
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'neg': scores[0], 
        'neu': scores[1], 
        'pos': scores[2], 
    }
    return scores_dict

def getScore(headlines): 
    res =[]
    for i in headlines: 
        res.append(polarity_scores(i))
    data = pd.DataFrame(res) 
    greaterPos = data['pos'] > 0 
    x = data['pos'][greaterPos].mean() 

    greaterNeg = data['neg'] > 0 
    y = data['neg'][greaterNeg].mean() 
    if pd.isna(x): 
        x=0 
    if pd.isna(y): 
        y=0

    return (x, y)
