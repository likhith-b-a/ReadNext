"""
Offline evaluation for the search/recommendation pipeline (utils/util_model.py).

ReadNext is an unsupervised, content-based search: there are no user click or
purchase logs to compute classic accuracy/precision against. Instead this
script measures whether the TF-IDF + cosine-similarity search reliably finds
the "obviously correct" book when given queries derived from that book's own
content -- without relying on the Category/Author metadata fields, since
those are already baked into the model's own feature weighting and would
make the check circular.

Three retrieval tests, each answering a different question about "does
search actually work":

1. Self-Retrieval@K       -- query = the book's own full summary.
                             Sanity check on the search plumbing itself
                             (vectorization, indexing, similarity math).
                             Expected to be very high; a drop here signals a
                             pipeline bug, not a relevance problem.

2. Masked-Retrieval@K     -- query = only the second half of the summary.
                             Tests whether the model still finds the right
                             book from partial information, not just an
                             exact-text match.

3. Structured-Keyword-Retrieval@K -- query = the book's own top-N highest-
                             weighted TF-IDF terms (its "keyword signature"),
                             e.g. "kitchen amy chinese god wife". This is the
                             closest proxy to how a real user actually
                             searches (a handful of keywords, not a
                             paragraph). Run for N=3 and N=5 to show how
                             score degrades as the query gets shorter/less
                             specific.

All three check whether the query book's own title appears in the top-K
results returned by utils.util_model.search_books_by_content.

Usage:
    venv/Scripts/python.exe model/evaluate_search.py
"""
import pickle
import random
import sys
import time

sys.path.insert(0, '.')
import utils.util_model as recommender

K = 3
SAMPLE_SIZE = 2000
SEED = 42
MIN_WORDS_SELF = 3
MIN_WORDS_MASKED = 10
KEYWORD_COUNTS = [3, 5]

# Field separator token from model/book_recommender.py's FIELD_SEP. It stays
# in the TF-IDF vocabulary (removing it as a stop word would undo the
# cross-field bigram fix -- see book_recommender.py), so it can surface as a
# top-weighted term for a document. It carries no meaning on its own, so it's
# excluded when picking a document's representative keywords (test 3 only).
SEP_TOKEN = 'xsepx'


def load_model():
    with open('model/model.pkl', 'rb') as f:
        return pickle.load(f)


def self_retrieval_score(df, tfidf, tfidf_matrix, indices):
    hits, total, skipped = 0, 0, 0
    for i in indices:
        book = df.iloc[i]
        query = book['processed_summary']
        if not isinstance(query, str) or len(query.split()) < MIN_WORDS_SELF:
            skipped += 1
            continue
        total += 1
        recs = recommender.search_books_by_content(query, tfidf, tfidf_matrix, df, top_n=K)
        if recs is not None and book['book_title'] in recs['book_title'].values:
            hits += 1
    return hits, total, skipped


def masked_retrieval_score(df, tfidf, tfidf_matrix, indices):
    hits, total, skipped = 0, 0, 0
    for i in indices:
        book = df.iloc[i]
        summary = book['processed_summary']
        if not isinstance(summary, str):
            skipped += 1
            continue
        words = summary.split()
        if len(words) < MIN_WORDS_MASKED:
            skipped += 1
            continue
        half = words[len(words) // 2:]
        query = ' '.join(half)
        total += 1
        recs = recommender.search_books_by_content(query, tfidf, tfidf_matrix, df, top_n=K)
        if recs is not None and book['book_title'] in recs['book_title'].values:
            hits += 1
    return hits, total, skipped


def structured_keyword_retrieval_score(df, tfidf, tfidf_matrix, indices, num_keywords):
    hits, total, skipped = 0, 0, 0
    feature_names = tfidf.get_feature_names_out()
    is_meaningful = [SEP_TOKEN not in name.split() for name in feature_names]
    for i in indices:
        book = df.iloc[i]
        row = tfidf_matrix[i].toarray().ravel()
        if row.sum() == 0:
            skipped += 1
            continue
        ranked = row.argsort()[::-1]
        top_idx = [j for j in ranked if row[j] > 0 and is_meaningful[j]][:num_keywords]
        if not top_idx:
            skipped += 1
            continue
        keywords = ' '.join(feature_names[j] for j in top_idx)
        total += 1
        recs = recommender.search_books_by_content(keywords, tfidf, tfidf_matrix, df, top_n=K)
        if recs is not None and book['book_title'] in recs['book_title'].values:
            hits += 1
    return hits, total, skipped


def main():
    model_data = load_model()
    df = model_data['books_df']
    tfidf = model_data['tfidf_vectorizer']
    tfidf_matrix = model_data['tfidf_matrix']

    random.seed(SEED)
    sample_idx = random.sample(range(len(df)), SAMPLE_SIZE)

    print(f"Sample size: {SAMPLE_SIZE} (seed={SEED}), K={K}\n")

    t0 = time.time()
    hits, total, skipped = self_retrieval_score(df, tfidf, tfidf_matrix, sample_idx)
    t1 = time.time()
    print(f"1) Self-Retrieval@{K}  (full summary as query)")
    print(f"   hits: {hits}/{total}  skipped: {skipped} (too short)")
    print(f"   score: {hits/total*100:.2f}%")
    print(f"   time: {t1-t0:.1f}s\n")

    hits, total, skipped = masked_retrieval_score(df, tfidf, tfidf_matrix, sample_idx)
    t2 = time.time()
    print(f"2) Masked-Retrieval@{K}  (second half of summary as query)")
    print(f"   hits: {hits}/{total}  skipped: {skipped} (too short to split)")
    print(f"   score: {hits/total*100:.2f}%")
    print(f"   time: {t2-t1:.1f}s\n")

    prev_t = t2
    for n in KEYWORD_COUNTS:
        hits, total, skipped = structured_keyword_retrieval_score(df, tfidf, tfidf_matrix, sample_idx, n)
        t_now = time.time()
        print(f"3) Structured-Keyword-Retrieval@{K}  (top-{n} TF-IDF terms as query)")
        print(f"   hits: {hits}/{total}  skipped: {skipped} (empty vector)")
        print(f"   score: {hits/total*100:.2f}%")
        print(f"   time: {t_now-prev_t:.1f}s\n")
        prev_t = t_now


if __name__ == "__main__":
    main()
