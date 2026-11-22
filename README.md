# IIS1_Semester_Short_Report
Repository containing our solution for the semester short report for the module "Intelligent Information Systems" at ZHAW. The script represents queries and documents as character bigram vectors and ranks documents by cosine similarity to each query.

## Setup and usage

Run these commands from the repository directory:

```sh
python -m pip install numpy
python bigram_search.py iis1_collection.trec iis1_queries.trec
```

Pass the collection file first and the query file second. The files can be stored anywhere. Put paths containing spaces in quotes.

Both input files use an XML structure in which each `<DOC>` contains a `<recordId>` and a `<text>` element. The script reads each record's ID and text from the XML.

## How it works

1. Replace punctuation and other characters outside the supported alphabet with spaces, convert the text to lowercase, and remove English stopwords. The stopword list is defined as `STOPWORDS` in `bigram_search.py`.

2. Count adjacent character pairs within each remaining word. The first character selects the row and the second selects the column in a 36 × 36 matrix. Both axes use `abcdefghijklmnopqrstuvwxyz0123456789`.

3. Apply L2 normalization and flatten the matrix row by row into a vector with 1'296 entries (36 x 36).

4. Compare each query vector with the document vectors using cosine similarity and rank the documents from highest to lowest score.

## Output files

The script writes the output to `results/` relative to the directory from which you run it. A new run overwrites these files in the selected output directory.

| File | Contents |
| --- | --- |
| `rankings.csv` | Query ID, rank, document ID, and cosine similarity for the top ten documents per query by default |
| `collection_vectors.npz` | Document IDs and their normalized bigram vectors |
| `queries_vectors.npz` | Query IDs and their normalized bigram vectors |

Use `--output-dir <directory>` to change the destination and `--top-k <number>` to change the number of results per query.

## Reading the vectors

An `.npz` file is a compressed archive of NumPy arrays. Each archive contains `record_ids`, `vectors`, and `alphabet`. The ID at index `i` belongs to row `i` in `vectors`. 




