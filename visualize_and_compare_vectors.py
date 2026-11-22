"""Zeige Query- und Dokumentvektoren als Bigramm-Heatmaps."""

import argparse
import csv
from pathlib import Path
from textwrap import wrap
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


def load_vectors(path):
    """Lade das Alphabet und ordne jeder Record-ID ihren Vektor zu."""
    with np.load(path) as archive:
        alphabet = str(archive["alphabet"].item())
        record_ids = archive["record_ids"]
        vectors = archive["vectors"]
    return alphabet, dict(zip(record_ids, vectors))


def read_rankings(path, query_id, max_rank):
    """Lies die gewünschten Query-Dokument-Paare aus der Ranking-Datei."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if query_id is not None and row["query_record_id"] != query_id:
                continue
            if int(row["rank"]) <= max_rank:
                yield row


def load_texts(path, record_ids):
    """Lies die Originaltexte der benötigten Record-IDs aus der TREC-Datei."""
    root = ET.parse(path).getroot()
    texts = {}
    for record in root.findall("DOC"):
        record_id = record.findtext("recordId", "").strip()
        if record_id in record_ids:
            text = record.find("text")
            if text is not None:
                original_text = "".join(text.itertext())
                texts[record_id] = " ".join(original_text.split())
    return texts


def draw_heatmap(axis, matrix, alphabet, title, max_value):
    """Zeichne eine Matrix mit Zeichenbeschriftung auf beiden Achsen."""
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max_value)
    positions = range(len(alphabet))
    axis.set_xticks(positions, labels=list(alphabet), fontsize=7)
    axis.set_yticks(positions, labels=list(alphabet), fontsize=7)
    axis.set_xlabel("Second character")
    axis.set_ylabel("First character")
    axis.set_title(title)
    return image


def save_comparison(path, row, query_vector, document_vector, alphabet, query_text, document_text):
    """Speichere beide Matrizen, ihre Originaltexte und den Ranking-Score."""
    size = len(alphabet)
    query_matrix = query_vector.reshape(size, size)
    document_matrix = document_vector.reshape(size, size)
    max_value = max(float(query_matrix.max()), float(document_matrix.max()))
    if max_value == 0:
        max_value = 1

    query_lines = wrap(query_text, width=64)
    document_lines = wrap(document_text, width=64)
    text_height = 0.35 + 0.16 * max(len(query_lines), len(document_lines))
    figure = plt.figure(figsize=(12, 6 + text_height))
    layout = figure.add_gridspec(
        3, 3, width_ratios=[1, 1, 0.045], height_ratios=[4.6, text_height, 0.3],
        left=0.08, right=0.92, bottom=0.04, top=0.94, hspace=0.30, wspace=0.27,
    )
    query_axis = figure.add_subplot(layout[0, 0])
    document_axis = figure.add_subplot(layout[0, 1])
    query_title = f"Query {row['query_record_id']}"
    document_title = f"Document {row['document_record_id']}"
    draw_heatmap(query_axis, query_matrix, alphabet, query_title, max_value)
    image = draw_heatmap(document_axis, document_matrix, alphabet, document_title, max_value)

    colorbar_axis = figure.add_subplot(layout[0, 2])
    figure.colorbar(image, cax=colorbar_axis, label="Normalized bigram weight")

    for column, title, lines in (
        (0, "Query text", query_lines),
        (1, "Collection document text", document_lines),
    ):
        text_axis = figure.add_subplot(layout[1, column])
        text_axis.set_axis_off()
        text_axis.text(0, 1, title, va="top", fontsize=10, fontweight="bold")
        text_axis.annotate(
            "\n".join(lines), xy=(0, 1), xycoords="axes fraction",
            xytext=(0, -16), textcoords="offset points",
            va="top", fontsize=8, family="monospace", parse_math=False,
        )

    score = row["cosine_similarity"]
    caption = f"Cosine similarity: {score}   |   Rank: {row['rank']}"
    caption_axis = figure.add_subplot(layout[2, :2])
    caption_axis.set_axis_off()
    caption_axis.text(0.5, 0.5, caption, ha="center", va="center", fontsize=13)
    try:
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--queries", type=Path, default=Path("iis1_queries.trec"))
    parser.add_argument("--collection", type=Path, default=Path("iis1_collection.trec"))
    parser.add_argument("--query-id", help="Nur diese Query anzeigen")
    parser.add_argument("--max-rank", type=int, default=1, help="Höchster Rang pro Query")
    args = parser.parse_args()
    if args.max_rank < 1:
        parser.error("--max-rank muss mindestens 1 sein")

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "plots"
    query_alphabet, query_vectors = load_vectors(input_dir / "queries_vectors.npz")
    document_alphabet, document_vectors = load_vectors(input_dir / "collection_vectors.npz")
    if query_alphabet != document_alphabet:
        parser.error("Die beiden Vektordateien verwenden unterschiedliche Alphabete")

    rankings = list(read_rankings(input_dir / "rankings.csv", args.query_id, args.max_rank))
    if not rankings:
        parser.error("Keine passenden Einträge in rankings.csv gefunden")

    query_ids = {row["query_record_id"] for row in rankings}
    document_ids = {row["document_record_id"] for row in rankings}
    try:
        query_texts = load_texts(args.queries, query_ids)
        document_texts = load_texts(args.collection, document_ids)
    except (OSError, ET.ParseError) as error:
        parser.error(f"Originaltexte konnten nicht geladen werden: {error}")
    for source, record_ids, texts in (
        (args.queries, query_ids, query_texts),
        (args.collection, document_ids, document_texts),
    ):
        missing_ids = record_ids.difference(texts)
        if missing_ids:
            parser.error(f"Fehlende Texte in {source}: {', '.join(sorted(missing_ids))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rankings:
        query_id = row["query_record_id"]
        document_id = row["document_record_id"]
        if query_id not in query_vectors or document_id not in document_vectors:
            parser.error(f"Vektor für Query {query_id} oder Dokument {document_id} fehlt")
        filename = f"query_{query_id}_rank_{row['rank']}_doc_{document_id}.png"
        save_comparison(
            output_dir / filename,
            row,
            query_vectors[query_id],
            document_vectors[document_id],
            query_alphabet,
            query_texts[query_id],
            document_texts[document_id],
        )
    print(f"{len(rankings)} Grafiken gespeichert in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
