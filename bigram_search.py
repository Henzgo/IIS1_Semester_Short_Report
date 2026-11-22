"""Erzeuge Bigramm-Vektoren aus XML-Dateien und vergleiche sie."""

import argparse
import csv
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
CHAR_INDEX = {char: index for index, char in enumerate(ALPHABET)}
MATRIX_SIZE = len(ALPHABET)
NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]+")
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from
had has have having he her here him his how i if in into is it its may me
might more most must my no nor not of on once only or other our out over own
same she should so some such than that the their them then there these they
this those through to too under until up us very was we were what when where
which while who whom why will with would you your
""".split())


def clean_text(text):
    """Ersetze Satzzeichen und andere Zeichen ausserhalb des Alphabets durch Leerzeichen."""
    return NON_ALPHANUMERIC.sub(" ", text).strip()


def lowercase_text(text):
    return text.lower()


def remove_stopwords(text, stopwords=STOPWORDS):
    words = text.split()
    remaining_words = [word for word in words if word not in stopwords]
    return " ".join(remaining_words)


def preprocessing(text):
    """Führe die drei Schritte der Textaufbereitung nacheinander aus."""
    cleaned_text = clean_text(text)
    lowercase = lowercase_text(cleaned_text)
    text_without_stopwords = remove_stopwords(lowercase)
    return text_without_stopwords


def text_to_bigram_matrix(text):
    """Zähle Bigramme innerhalb von Wörtern und normalisiere die Matrix."""
    prepared_text = preprocessing(text)
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for word in prepared_text.split():
        for first, second in zip(word, word[1:]):
            row = CHAR_INDEX[first]
            column = CHAR_INDEX[second]
            matrix[row, column] += 1
    norm = np.linalg.norm(matrix)
    if norm:
        matrix /= norm
    return matrix


def matrix_to_vector(matrix):
    """Lies die 36×36-Matrix Zeile für Zeile als Vektor mit 1296 Werten."""
    return np.asarray(matrix).reshape(MATRIX_SIZE ** 2)


def read_trec(path):
    """Lies ID und Text aus jeder <DOC>-Gruppe der TREC-XML-Datei."""
    root = ET.parse(path).getroot()
    for document in root.findall("DOC"):
        record_id = document.findtext("recordId", "").strip()
        text = document.find("text")
        if not record_id or text is None:
            raise ValueError(f"Missing recordId or text in {path}")
        yield record_id, "".join(text.itertext())


def vectorize_trec(path):
    """Erzeuge pro ID einen Vektor an derselben Position im Vektor-Array."""
    record_ids = []
    vectors = []
    for record_id, text in read_trec(path):
        matrix = text_to_bigram_matrix(text)
        vector = matrix_to_vector(matrix)
        record_ids.append(record_id)
        vectors.append(vector)
    return (
        np.asarray(record_ids, dtype=str),
        np.asarray(vectors, dtype=np.float32).reshape(-1, MATRIX_SIZE ** 2),
    )


def search(query_vector, document_vectors):
    """Gib (Dokumentindex, Cosine Similarity) absteigend nach Score zurück."""
    query_vector = np.asarray(query_vector, dtype=np.float64)
    document_vectors = np.asarray(document_vectors, dtype=np.float64)
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    dot_products = document_vectors @ query_vector
    denominators = query_norm * document_norms
    # Bei einem Nullvektor bleibt der Score 0.
    scores = np.divide(
        dot_products,
        denominators,
        out=np.zeros(len(document_vectors)),
        where=denominators != 0,
    )
    scores = np.clip(scores, -1, 1)
    ranked_indices = np.argsort(-scores, kind="stable")
    return [(int(index), float(scores[index])) for index in ranked_indices]


def save_vectors(path, record_ids, vectors):
    np.savez_compressed(
        path, record_ids=record_ids, vectors=vectors, alphabet=ALPHABET
    )


def write_rankings(path, query_ids, query_vectors, document_ids, document_vectors, top_k):
    document_vectors = document_vectors.astype(np.float64)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query_record_id", "rank", "document_record_id", "cosine_similarity"])
        for query_id, query_vector in zip(query_ids, query_vectors):
            ranking = search(query_vector, document_vectors)
            for position, (index, score) in enumerate(ranking[:top_k], start=1):
                document_id = document_ids[index]
                writer.writerow([query_id, position, document_id, f"{score:.9f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    document_ids, document_vectors = vectorize_trec(args.collection)
    query_ids, query_vectors = vectorize_trec(args.queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_vectors(args.output_dir / "collection_vectors.npz", document_ids, document_vectors)
    save_vectors(args.output_dir / "queries_vectors.npz", query_ids, query_vectors)
    write_rankings(
        args.output_dir / "rankings.csv",
        query_ids,
        query_vectors,
        document_ids,
        document_vectors,
        args.top_k,
    )
    print(f"Processed {len(document_ids)} documents and {len(query_ids)} queries.")


if __name__ == "__main__":
    main()
