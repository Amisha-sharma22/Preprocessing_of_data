import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import json
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# nltk.download('all')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text)
    words = word_tokenize(text.lower())
    pattern = re.compile("^[a-zA-Z0-9+\-^*/%=]+$")
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and pattern.match(word) ]
    return ' '.join(words)

with open('data.json', 'r') as file:
    data = json.load(file)

questions = []
answers = []

for subject, qas in data.items():
    for qa in qas:
        questions.append(qa['Question'])
        answers.append(qa['Answer'])

df = pd.DataFrame({'question': questions, 'answer': answers})

df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)


df.to_csv('processed_data.csv', index=False)

print(df)
