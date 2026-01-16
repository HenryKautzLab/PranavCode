#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import orjson

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer

import igraph as ig

try:
    import leidenalg
    HAVE_LEIDEN = True
except Exception:
    HAVE_LEIDEN = False

# optional
try:
    from nltk.corpus import stopwords as nltk_stopwords
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

try:
    from wordcloud import WordCloud
    HAVE_WORDCLOUD = True
except Exception:
    HAVE_WORDCLOUD = False


def load_rows(path: Path):
    rows = []
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def build_knn_graph(X: np.ndarray, k: int):
    """
    Build an undirected weighted kNN graph using cosine similarity.
    Edge weight = max(sim(i->j), sim(j->i)) to avoid duplicates.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    edges = {}
    n = X.shape[0]
    for i in range(n):
        for j, d in zip(idxs[i], dists[i]):
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            sim = 1.0 - float(d)
            edges[(a, b)] = max(edges.get((a, b), 0.0), sim)

    g = ig.Graph(n=n)
    g.add_edges(list(edges.keys()))
    g.es["weight"] = list(edges.values())
    return g


def cluster_graph(g: ig.Graph, seed: int):
    """
    Leiden (if installed) else igraph multilevel (Louvain-like).
    """
    if HAVE_LEIDEN:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=seed
        )
        return np.array(part.membership, dtype=int)

    part = g.community_multilevel(weights="weight")
    return np.array(part.membership, dtype=int)


def get_stopwords(extra: bool = True):
    base = set(ENGLISH_STOP_WORDS)
    if HAVE_NLTK:
        try:
            base |= set(nltk_stopwords.words("english"))
        except Exception:
            pass

    if extra:
        base |= {
            # reddit / mod boilerplate
            "removed", "please", "subreddit", "submission", "moderators", "mods",
            "automatically", "bot", "compose", "performed", "message", "contact",
            "concerns", "comment", "comments", "post", "posts", "rule", "rules",
            "thread", "flair", "flaired", "approved", "unverified",
            # filler/junk
            "lol", "lmao", "imo", "idk", "im", "ive", "dont", "cant", "wont",
            "yeah", "ok", "okay", "thanks", "thank", "u", "ur",
            # artifacts
            "http", "https", "www", "com", "reddit", "amp",
        }

    return list(base)


def normalize_text_for_tfidf(s: str) -> str:
    s = s or ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_cluster_tfidf_terms(
    df: pd.DataFrame,
    text_col: str,
    cluster_col: str,
    stopwords,
    min_df: int,
    max_df_ratio: float,
    max_features: int,
    ngram_max: int,
    top_terms: int,
):
    """
    Two-pass approach:
    1) CountVectorizer to compute doc frequency ratio across ALL texts.
       Keep vocab terms that appear in <= max_df_ratio of docs.
    2) TfidfVectorizer with that filtered vocab + stopwords.
    Then: sum TF-IDF within each cluster to get top terms.
    """
    texts = df[text_col].fillna("").astype(str).map(normalize_text_for_tfidf).tolist()
    if len(texts) == 0:
        return {}, {}

    # Pass 1: DF ratio filter
    count_vec = CountVectorizer(min_df=1, stop_words=None)
    Xc = count_vec.fit_transform(texts)
    terms_all = np.array(count_vec.get_feature_names_out())

    N = Xc.shape[0]
    df_ratio = (Xc > 0).sum(axis=0).A1 / max(N, 1)

    keep_mask = df_ratio <= max_df_ratio
    vocab = terms_all[keep_mask]
    if vocab.size == 0:
        vocab = terms_all  # fallback

    # Pass 2: TF-IDF
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        stop_words=stopwords,
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
    )
    X = vectorizer.fit_transform(texts)
    tf_terms = np.array(vectorizer.get_feature_names_out())

    # Per-cluster top terms
    cluster_terms = {}
    cluster_labels = {}

    clusters = sorted(df[cluster_col].unique().tolist())
    cluster_ids = df[cluster_col].astype(int).values

    for c in clusters:
        idx = np.where(cluster_ids == int(c))[0]
        if idx.size == 0:
            cluster_terms[c] = []
            cluster_labels[c] = f"Cluster {c}"
            continue

        scores = X[idx].sum(axis=0).A1
        if scores.max() <= 0:
            cluster_terms[c] = []
            cluster_labels[c] = f"Cluster {c}"
            continue

        top_idx = scores.argsort()[::-1][:top_terms]
        top = [tf_terms[i] for i in top_idx]
        cluster_terms[c] = top
        cluster_labels[c] = ", ".join(top[:3]) if len(top) >= 3 else (", ".join(top) if top else f"Cluster {c}")

    return cluster_terms, cluster_labels


def write_wordclouds(out_dir: Path, cluster_terms: dict, cluster_texts: dict, max_words: int = 200):
    if not HAVE_WORDCLOUD:
        print("NOTE: wordcloud not installed; skipping wordcloud PNGs.")
        return

    wc_dir = out_dir / "wordclouds"
    wc_dir.mkdir(parents=True, exist_ok=True)

    for c, texts in cluster_texts.items():
        blob = " ".join(texts)
        if not blob.strip():
            continue
        wc = WordCloud(width=1400, height=900, max_words=max_words, background_color="white")
        img = wc.generate(blob).to_image()
        img.save(wc_dir / f"cluster_{c}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--max-items", type=int, default=8000)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--knn-k", type=int, default=30)
    ap.add_argument("--sample-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # TF-IDF labeling config
    ap.add_argument("--tfidf-min-df", type=int, default=10)
    ap.add_argument("--tfidf-max-df-ratio", type=float, default=0.05)
    ap.add_argument("--tfidf-max-features", type=int, default=20000)
    ap.add_argument("--tfidf-ngram-max", type=int, default=3)
    ap.add_argument("--top-terms", type=int, default=20)

    # Gephi export
    ap.add_argument("--export-gephi", action="store_true", help="Write GraphML + nodes/edges CSV for Gephi")

    # Wordclouds
    ap.add_argument("--wordclouds", action="store_true", help="Write wordcloud PNGs per cluster (requires wordcloud)")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.input_jsonl))

    # pool: MAHA or high confidence
    pool = []
    for r in rows:
        lab = str(r.get("maha_label", "")).upper()
        conf = float(r.get("maha_confidence", 0.0) or 0.0)

        # your original logic: keep MAHA OR conf>=min_conf
        if lab == "MAHA" or conf >= args.min_conf:
            txt = (r.get("text") or "").strip()
            if txt:
                pool.append(r)

    random.shuffle(pool)
    pool = pool[:args.max_items]
    if len(pool) < 200:
        raise SystemExit(f"Pool too small ({len(pool)}). Lower --min-conf or increase --max-items.")

    # text truncation for embedding
    texts = [p["text"][:2000] for p in pool]

    emb = SentenceTransformer(args.embed_model)
    X = emb.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    g = build_knn_graph(X, k=args.knn_k)
    labels = cluster_graph(g, seed=args.seed)

    df = pd.DataFrame(pool)
    df["cluster"] = labels

    # robust columns
    if "maha_label" not in df.columns:
        df["maha_label"] = ""
    if "maha_confidence" not in df.columns:
        df["maha_confidence"] = 0.0

    # ===== TF-IDF TOP TERMS PER CLUSTER =====
    stopwords = get_stopwords(extra=True)
    cluster_terms, cluster_labels = compute_cluster_tfidf_terms(
        df=df,
        text_col="text",
        cluster_col="cluster",
        stopwords=stopwords,
        min_df=args.tfidf_min_df,
        max_df_ratio=args.tfidf_max_df_ratio,
        max_features=args.tfidf_max_features,
        ngram_max=args.tfidf_ngram_max,
        top_terms=args.top_terms,
    )

    # Save cluster top terms table
    terms_rows = []
    for c in sorted(df["cluster"].unique()):
        top = cluster_terms.get(int(c), [])
        terms_rows.append({
            "cluster": int(c),
            "label": cluster_labels.get(int(c), f"Cluster {c}"),
            "top_terms": ", ".join(top),
        })
    df_terms = pd.DataFrame(terms_rows)
    df_terms.to_csv(out_dir / "cluster_top_terms.csv", index=False)

    # ===== SUMMARY =====
    summary = (
        df.groupby("cluster")
          .agg(
              n=("cluster", "size"),
              mean_conf=("maha_confidence", "mean"),
              frac_maha=("maha_label", lambda s: (s.astype(str).str.upper() == "MAHA").mean()),
          )
          .reset_index()
    )
    summary["label"] = summary["cluster"].map(lambda c: cluster_labels.get(int(c), f"Cluster {c}"))
    summary = summary.merge(df_terms[["cluster", "top_terms"]], on="cluster", how="left")
    summary = summary.sort_values(["n", "mean_conf"], ascending=False)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    # ===== SAMPLE ACROSS CLUSTERS (QUOTA STRATIFIED) =====
    # good_clusters = summary[summary["mean_conf"] >= args.min_conf]["cluster"].tolist()
    # if not good_clusters:
    #     good_clusters = summary["cluster"].tolist()

    # C = len(good_clusters)
    # per = max(1, int(np.ceil(args.sample_n / C)))
    #
    # samples = []
    # for c in good_clusters:
    #     sub = df[df["cluster"] == c]
    #     if sub.empty:
    #         continue
    #     take = min(per, len(sub))
    #     samples.append(sub.sample(n=take, random_state=args.seed))
    #
    # sampled = pd.concat(samples, axis=0) if samples else df.sample(n=min(args.sample_n, len(df)), random_state=args.seed)
    # sampled = sampled.sample(n=min(args.sample_n, len(sampled)), random_state=args.seed)
    #
    # # add empty gold columns for annotation
    # sampled = sampled.copy()
    # sampled["gold_maha"] = ""
    # sampled["gold_theme"] = ""        # single label
    # sampled["gold_stance"] = ""
    # sampled["gold_sarcasm"] = ""
    # sampled["gold_discourse"] = ""
    #
    # sampled.to_csv(out_dir / "sample_for_annotation.csv", index=False)

    # ===== SAMPLE PER CLUSTER (100 FROM EACH CLUSTER) =====
    # New args you should add:
    # ap.add_argument("--sample-per-cluster", type=int, default=100)

    samples = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        if sub.empty:
            continue
        take = min(args.sample_per_cluster, len(sub))
        samp = sub.sample(n=take, random_state=args.seed)
        samp = samp.copy()
        samp["cluster"] = int(c)
        samples.append(samp)

    sampled = pd.concat(samples, axis=0) if samples else df.head(0)

    # add empty gold columns for annotation
    sampled = sampled.copy()
    sampled["gold_maha"] = ""
    sampled["gold_theme"] = ""
    sampled["gold_stance"] = ""
    sampled["gold_sarcasm"] = ""
    sampled["gold_discourse"] = ""

    sampled.to_csv(out_dir / "sample_for_annotation.csv", index=False)
    print(f"Sampled {len(sampled)} rows = sum_c min(100, |cluster_c|)")

    # ===== GEPHI EXPORT =====
    if args.export_gephi:
        # Create node table
        node_df = pd.DataFrame({
            "Id": np.arange(len(df)),
            "Label": df["text"].fillna("").astype(str).map(lambda s: s[:60].replace("\n", " ")),
            "cluster": df["cluster"].astype(int),
            "cluster_label": df["cluster"].astype(int).map(lambda c: cluster_labels.get(int(c), f"Cluster {c}")),
            "maha_label": df["maha_label"].astype(str),
            "maha_confidence": df["maha_confidence"].astype(float),
        })
        node_df.to_csv(out_dir / "gephi_nodes.csv", index=False)

        # Edge table
        edgelist = []
        for e in g.es:
            s = int(e.tuple[0])
            t = int(e.tuple[1])
            w = float(e["weight"])
            edgelist.append((s, t, w))
        edge_df = pd.DataFrame(edgelist, columns=["Source", "Target", "Weight"])
        edge_df.to_csv(out_dir / "gephi_edges.csv", index=False)

        # Also write GraphML with node attributes
        g.vs["cluster"] = df["cluster"].astype(int).tolist()
        g.vs["cluster_label"] = df["cluster"].astype(int).map(lambda c: cluster_labels.get(int(c), f"Cluster {c}")).tolist()
        g.vs["maha_label"] = df["maha_label"].astype(str).tolist()
        g.vs["maha_confidence"] = df["maha_confidence"].astype(float).tolist()

        g.write_graphml(str(out_dir / "knn_graph.graphml"))

    # ===== WORDCLOUDS =====
    if args.wordclouds:
        # provide per-cluster texts for wordcloud
        cluster_texts = {}
        for c in sorted(df["cluster"].unique()):
            cluster_texts[int(c)] = df.loc[df["cluster"] == c, "text"].fillna("").astype(str).tolist()
        write_wordclouds(out_dir, cluster_terms, cluster_texts)

    print(f"WROTE: {out_dir/'cluster_summary.csv'}")
    print(f"WROTE: {out_dir/'cluster_top_terms.csv'}")
    print(f"WROTE: {out_dir/'sample_for_annotation.csv'}")
    if args.export_gephi:
        print(f"WROTE: {out_dir/'gephi_nodes.csv'}")
        print(f"WROTE: {out_dir/'gephi_edges.csv'}")
        print(f"WROTE: {out_dir/'knn_graph.graphml'}")
    if args.wordclouds and HAVE_WORDCLOUD:
        print(f"WROTE: {out_dir/'wordclouds'}/*.png")
    print(f"Clusters: {df['cluster'].nunique()}  Pool: {len(df)}")


if __name__ == "__main__":
    main()
