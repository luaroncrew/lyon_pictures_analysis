from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd

stop_words = stopwords.words('english')
stop_words.extend(stopwords.words('french'))
stop_words.extend(stopwords.words('italian'))
stop_words.extend(stopwords.words('spanish'))
stop_words.extend(stopwords.words('portuguese'))
stop_words.extend(stopwords.words('german'))
stop_words.extend(
    [
        "france",
        "lyon",
        "69",
        "rhônealpes",
        "french",
        "français",
        "ville",
    ]
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
    initial_data['text_data'] = initial_data[' title'].fillna('')
    initial_data['cleaned_text'] = initial_data['text_data'].apply(clean_and_lemmatize)

    initial_data.to_csv("lemmatized_data.csv")




