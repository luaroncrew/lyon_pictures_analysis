from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def translate_clusters_to_text_documents(data):
    df = data.dropna(subset=["cluster", "cleaned_text"])
    cluster_documents = (
        df.groupby("cluster")["cleaned_text"]
        .apply(lambda texts: " ".join(texts))
        .reset_index()
        .rename(columns={"cleaned_text": "cluster_text"})
    )
    return cluster_documents


def extract_cluster_keywords_from_documents(
        cluster_docs: pd.DataFrame,
        top_n: int = 10,
        max_document_frequency: float = 0.8,
        ngram_range: tuple = (1, 2)
) -> pd.DataFrame:
    documents = cluster_docs["cluster_text"].tolist()
    cluster_ids = cluster_docs["cluster"].tolist()

    vectorizer = TfidfVectorizer(
        max_df=max_document_frequency, min_df=2, ngram_range=ngram_range
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    top_terms_per_cluster = []

    # Loop through each cluster's TF-IDF vector
    for idx, row in enumerate(tfidf_matrix):
        row_data = row.toarray().flatten()
        top_indices = row_data.argsort()[::-1][:top_n]
        top_features = [(feature_names[i], row_data[i]) for i in top_indices]

        top_terms_per_cluster.append({
            "cluster": cluster_ids[idx],
            "top_terms": [term for term, score in top_features]
        })

    # Convert to DataFrame for display
    top_terms_df = pd.DataFrame(top_terms_per_cluster)

    return top_terms_df


def get_clusters_keywords(data):
    docs = translate_clusters_to_text_documents(data)
    keywords = extract_cluster_keywords_from_documents(docs)
    return keywords


