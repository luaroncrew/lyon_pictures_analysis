from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd

stop_words = set(
    stopwords.words('english')
).update(
    set(stopwords.words('french'))
)
lemmatizer = WordNetLemmatizer()


def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords, and lemmatize
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return " ".join(cleaned)


if __name__ == '__main__':
    initial_data = pd.read_csv("initial_data.csv")



    initial_data = initial_data.head(10000)
    initial_data['text_data'] = initial_data[' title'].fillna('') + " " + initial_data[' tags'].fillna('')
    initial_data['cleaned_text'] = initial_data['text_data'].apply(clean_and_lemmatize)

    print(initial_data)


