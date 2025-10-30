#!/usr/bin/env python3
"""
hybrid_text_pipeline.py

Hybrid MPI + multiprocessing pipeline.

- MPI across nodes: scatter dataset chunks to ranks
- Within each MPI rank: spawn local worker processes to tokenize & count in parallel
- Sentiment filter:
    * rank 0: count only positive reviews (score >= 4)
    * rank 1: count only negative reviews (score <= 2)
    * other ranks: count all reviews
- Master (rank 0) aggregates results, reports per-node times, communication overhead,
  hybrid total time, and speedup vs sequential baseline.

Run:
  mpiexec -n 2 python hybrid_text_pipeline.py --max-rows 20000 --local-workers 4 --imbalance 0 --measure-seq

"""
from mpi4py import MPI
import csv
import re
import time
import argparse
import multiprocessing as mp
from collections import Counter

INPUT_FILE = "reviews.csv"
TOKEN_RE = re.compile(r"[a-z']+")
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as",
    "at","be","because","been","before","being","below","between","both","but","by","can't","cannot",
    "could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd",
    "he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i",
    "i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's",
    "should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them",
    "themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this",
    "those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll",
    "we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while",
    "who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves"
}

def detect_columns(header):
    lower = [h.strip().lower() for h in header]
    text_candidates = ("review", "text", "review_text", "summary", "comment", "body")
    for c in text_candidates:
        if c in lower:
            text_idx = lower.index(c)
            break
    else:
        # fallback: last column is usually text
        text_idx = len(lower) - 1

    # Score typical name
    if "score" in lower:
        score_idx = lower.index("score")
    elif "rating" in lower:
        score_idx = lower.index("rating")
    else:
        # fallback: look for numeric-like column; default to 6 if present
        score_idx = 6 if len(lower) > 6 else None
    return text_idx, score_idx

def load_rows(path, max_rows):
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return []
            text_idx, score_idx = detect_columns(header)
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                text = r[text_idx] if text_idx is not None and text_idx < len(r) else ""
                score = None
                if score_idx is not None and score_idx < len(r):
                    try:
                        score = int(float(r[score_idx]))
                    except Exception:
                        score = None
                rows.append((score, text))
    except FileNotFoundError:
        return []
    return rows

def clean_and_tokenize(text):
    if not text:
        return []
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]

# This must be top-level for pickling by multiprocessing
def local_worker_task(args):
    """
    args: (subrows, rank, sentiment_mode)
    subrows: list of (score, text)
    sentiment_mode: "positive" | "negative" | "all"
    returns: (counter_dict, assigned_count, kept_count, elapsed_seconds)
    assigned_count = number of rows processed by this worker (assigned to it)
    kept_count = number of rows that passed sentiment filter and had tokens
    """
    subrows, rank, sentiment_mode = args
    t0 = time.perf_counter()
    c = Counter()
    assigned = len(subrows)
    kept = 0
    for score, text in subrows:
        keep = True
        if sentiment_mode == "positive":
            keep = (score is not None and score >= 4)
        elif sentiment_mode == "negative":
            keep = (score is not None and score <= 2)
        if not keep:
            continue
        tokens = clean_and_tokenize(text)
        if tokens:
            c.update(tokens)
            kept += 1
    elapsed = time.perf_counter() - t0
    return dict(c), assigned, kept, elapsed

def chunkify(seq, n):
    if n <= 0:
        return []
    L = len(seq)
    base = L // n
    rem = L % n
    chunks = []
    i = 0
    for k in range(n):
        add = 1 if k < rem else 0
        end = i + base + add
        chunks.append(seq[i:end])
        i = end
    return chunks

def run_local_pool(local_rows, n_local_workers, rank, sentiment_mode):
    """
    Spawn a multiprocessing.Pool to process local_rows in parallel.
    Returns combined Counter, assigned_count, kept_count, node_elapsed (wall time)
    and detailed worker elapsed list.
    """
    if not local_rows:
        return Counter(), 0, 0, 0.0, []

    # allow n_local_workers == 0 (process sequentially in this rank)
    if n_local_workers <= 0:
        t0 = time.perf_counter()
        total_c = Counter()
        assigned = len(local_rows)
        kept = 0
        for score, text in local_rows:
            keep = True
            if sentiment_mode == "positive":
                keep = (score is not None and score >= 4)
            elif sentiment_mode == "negative":
                keep = (score is not None and score <= 2)
            if not keep:
                continue
            toks = clean_and_tokenize(text)
            if toks:
                total_c.update(toks)
                kept += 1
        elapsed = time.perf_counter() - t0
        return total_c, assigned, kept, elapsed, [elapsed]

    sublists = chunkify(local_rows, n_local_workers)
    args_iter = [(subl, rank, sentiment_mode) for subl in sublists]
    node_start = time.perf_counter()
    # On Windows Pool must be created under __main__ guard — main() is guarded below.
    with mp.Pool(processes=n_local_workers) as pool:
        results = pool.map(local_worker_task, args_iter)
    node_end = time.perf_counter()
    node_elapsed = node_end - node_start

    total_counter = Counter()
    total_assigned = 0
    total_kept = 0
    worker_times = []
    for counter_dict, assigned, kept, elapsed in results:
        total_counter.update(counter_dict)
        total_assigned += assigned
        total_kept += kept
        worker_times.append(elapsed)
    return total_counter, total_assigned, total_kept, node_elapsed, worker_times

def split_chunks(rows, n_ranks, imbalance_extra=0):
    total = len(rows)
    imbalance_extra = max(0, min(imbalance_extra, total))
    remaining = total - imbalance_extra
    base = remaining // n_ranks if n_ranks > 0 else 0
    rem = remaining % n_ranks if n_ranks > 0 else 0
    chunks = []
    idx = 0
    for r in range(n_ranks):
        size = base + (1 if r < rem else 0)
        if r == 0:
            size += imbalance_extra
        chunks.append(rows[idx: idx + size])
        idx += size
    if idx < total:
        chunks[-1].extend(rows[idx:])
    while len(chunks) < n_ranks:
        chunks.append([])
    return chunks

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid MPI + multiprocessing text pipeline")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--imbalance", type=int, default=0,
                   help="give rank 0 this many extra rows (imbalance)")
    # allow 0 local workers to run sequentially inside each MPI rank (minimizes cores)
    p.add_argument("--local-workers", type=int, default=max(0, mp.cpu_count() // 2),
                   help="number of local worker processes per MPI rank (0 = no pool, run sequentially)")
    p.add_argument("--measure-seq", action="store_true", help="measure sequential baseline")
    return p.parse_args()

def sequential_baseline(rows):
    t0 = time.perf_counter()
    pos = Counter()
    neg = Counter()
    for score, text in rows:
        toks = clean_and_tokenize(text)
        if not toks:
            continue
        if score is not None and score >= 4:
            pos.update(toks)
        if score is not None and score <= 2:
            neg.update(toks)
    elapsed = time.perf_counter() - t0
    return pos, neg, elapsed

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master loads and splits
    if rank == 0:
        rows = load_rows(INPUT_FILE, args.max_rows)
        chunks = split_chunks(rows, size, imbalance_extra=args.imbalance)
    else:
        chunks = None

    # start distributed timer (includes scatter, local compute, gather)
    dist_start = MPI.Wtime()
    local_rows = comm.scatter(chunks, root=0)

    # sentiment mode per rank
    if rank == 0:
        sentiment_mode = "positive"
    elif rank == 1:
        sentiment_mode = "negative"
    else:
        sentiment_mode = "all"

    # local parallel processing
    local_counter, assigned_count, kept_count, node_elapsed, worker_times = run_local_pool(
        local_rows, args.local_workers, rank, sentiment_mode
    )

    # prepare info to send back
    info = {
        "rank": rank,
        "assigned": assigned_count,
        "kept": kept_count,
        "node_time": node_elapsed,
        "worker_times": worker_times,
        "counter": dict(local_counter),
        "sentiment_mode": sentiment_mode
    }

    gathered = comm.gather(info, root=0)
    dist_end = MPI.Wtime()
    distributed_elapsed = dist_end - dist_start

    if rank == 0:
        # sort by rank
        gathered.sort(key=lambda x: x["rank"])
        # reconstruct counters
        total_counter = Counter()
        pos_counter = Counter()
        neg_counter = Counter()
        node_times = []
        total_assigned = 0
        total_kept_pos = 0
        total_kept_neg = 0
        for g in gathered:
            total_counter.update(g["counter"])
            node_times.append(g["node_time"])
            total_assigned += g["assigned"]
            if g["sentiment_mode"] == "positive":
                pos_counter.update(g["counter"])
                total_kept_pos += g["kept"]
            if g["sentiment_mode"] == "negative":
                neg_counter.update(g["counter"])
                total_kept_neg += g["kept"]

        # sequential baseline (optional)
        seq_time = None
        seq_pos = seq_neg = None
        if args.measure_seq:
            seq_pos, seq_neg, seq_time = sequential_baseline([r for ch in chunks for r in ch])

        # per-node print (use assigned counts and node_time)
        for g in gathered:
            label = f"Node {g['rank']}"
            if g["sentiment_mode"] == "positive":
                label += " (positive)"
            elif g["sentiment_mode"] == "negative":
                label += " (negative)"
            else:
                label += " (all)"
            # print assigned (total rows given to node) and kept (rows matching sentiment & tokens)
            print(f"● {label}: processed {g['assigned']:,} reviews (kept {g['kept']:,}) in {g['node_time']:.1f}s")

        # explicit totals for pos/neg
        print(f"● Total positive reviews kept (across positive nodes): {total_kept_pos:,}")
        print(f"● Total negative reviews kept (across negative nodes): {total_kept_neg:,}")

        # communication overhead estimate: distributed_elapsed minus max node compute time
        max_node_time = max(node_times) if node_times else 0.0
        comm_overhead = distributed_elapsed - max_node_time
        print(f"● Communication overhead: {comm_overhead:.1f}s")
        print(f"● Hybrid total time: {distributed_elapsed:.1f}s")

        if seq_time:
            speedup = seq_time / distributed_elapsed if distributed_elapsed > 0 else float('inf')
            print(f"● Hybrid speedup: {speedup:.2f}x vs Sequential ({seq_time:.1f}s)")

        # Most common sentiment words: top 5 for positive and negative
        top_pos = pos_counter.most_common(5)
        top_neg = neg_counter.most_common(5)
        if top_pos:
            top_pos_str = ", ".join(f"{w}({c})" for w, c in top_pos)
            print(f"● Top positive words: {top_pos_str}")
        else:
            print("● Top positive words: (none)")

        if top_neg:
            top_neg_str = ", ".join(f"{w}({c})" for w, c in top_neg)
            print(f"● Top negative words: {top_neg_str}")
        else:
            print("● Top negative words: (none)")

if __name__ == "__main__":
    main()