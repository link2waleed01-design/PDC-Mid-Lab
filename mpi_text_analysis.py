#!/usr/bin/env python3
"""
mpi_text_analysis.py

MPI-based text analysis using mpi4py.

Master (rank 0):
 - reads up to MAX_ROWS lines from INPUT_FILE
 - splits into `size` chunks (optional imbalance for rank 0)
 - scatters chunks to all ranks

All ranks:
 - receive chunk, perform cleaning + tokenization + local counting
 - send back local stats (processed, time, counter) via gather

Master aggregates and prints per-rank stats and total distributed time.
"""
from mpi4py import MPI
import csv
import re
import time
import argparse
from collections import Counter

INPUT_FILE = "reviews.csv"
MAX_ROWS = 20000

TOKEN_RE = re.compile(r"[a-z']+")  # keep letters and apostrophes

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

def detect_text_column(header):
    lower = [h.strip().lower() for h in header]
    candidates = ("review", "text", "review_text", "comment", "content", "body")
    for name in candidates:
        if name in lower:
            return lower.index(name)
    if len(lower) == 1:
        return 0
    for i, h in enumerate(lower):
        if not any(token in h for token in ("id", "rating", "score", "date")):
            return i
    return 0

def clean_and_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def worker_count(texts):
    c = Counter()
    for t in texts:
        c.update(clean_and_tokenize(t))
    return c

def split_chunks(texts, n_ranks, imbalance_extra=0):
    """Return list of `n_ranks` chunks. Rank 0 receives `imbalance_extra` extra lines."""
    total = len(texts)
    if n_ranks <= 0:
        return []
    # Ensure imbalance_extra isn't larger than total
    imbalance_extra = max(0, min(imbalance_extra, total))
    # Distribute remaining items evenly
    remaining = total - imbalance_extra
    base = remaining // n_ranks
    rem = remaining % n_ranks
    chunks = []
    idx = 0
    for r in range(n_ranks):
        size = base + (1 if r < rem else 0)
        if r == 0:
            size += imbalance_extra
        chunks.append(texts[idx: idx + size])
        idx += size
    # safety: append leftovers to last chunk
    if idx < total:
        chunks[-1].extend(texts[idx:])
    # ensure length == n_ranks
    while len(chunks) < n_ranks:
        chunks.append([])
    return chunks

def load_texts(path, max_rows):
    texts = []
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return texts
            text_idx = detect_text_column(header)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                texts.append(row[text_idx] if text_idx < len(row) else "")
    except FileNotFoundError:
        return texts
    return texts

def parse_args():
    p = argparse.ArgumentParser(description="MPI text analysis with mpi4py")
    p.add_argument("--max-rows", type=int, default=MAX_ROWS, help="max rows to read (default 20000)")
    p.add_argument("--imbalance", type=int, default=0,
                   help="give rank 0 this many extra lines (introduce imbalance)")
    p.add_argument("--measure-seq", action="store_true",
                   help="measure sequential baseline time (may double total runtime)")
    return p.parse_args()

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        texts = load_texts(INPUT_FILE, args.max_rows)
        chunks = split_chunks(texts, size, imbalance_extra=args.imbalance)
    else:
        chunks = None

    # distributed timer around scatter + local work + gather
    dist_start = MPI.Wtime()
    # scatter chunks: each rank receives its chunk
    local_chunk = comm.scatter(chunks, root=0)
    # local work
    local_start = MPI.Wtime()
    local_counter = worker_count(local_chunk)
    local_end = MPI.Wtime()
    local_info = {
        "rank": rank,
        "processed": len(local_chunk),
        "time": local_end - local_start,
        "counter": dict(local_counter)  # send as plain dict for safety
    }
    # gather results at root
    gathered = comm.gather(local_info, root=0)
    dist_end = MPI.Wtime()

    if rank == 0:
        distributed_elapsed = dist_end - dist_start
        # rebuild counters and sort by rank
        gathered.sort(key=lambda x: x["rank"])
        total_counter = Counter()
        for info in gathered:
            total_counter.update(info["counter"])

        # sequential baseline (single-process) -- measured AFTER distributed run
        # so it does not affect the distributed timing
        sstart = time.perf_counter()
        _ = worker_count([t for c in chunks for t in c])  # run full workload sequentially
        seq_time = time.perf_counter() - sstart

        # print per-rank stats
        for info in gathered:
            print(f"● Rank {info['rank']} processed {info['processed']:,} lines in {info['time']:.1f}s")
        print(f"● Total distributed time: {distributed_elapsed:.1f}s")
        # print speedup over sequential
        speedup = seq_time / distributed_elapsed if distributed_elapsed > 0 else float('inf')
        print(f"● Speedup over sequential: {speedup:.2f}x")

        # top words summary
        top = total_counter.most_common(5)
        top_str = ", ".join(f"{w}({c})" for w, c in top)
        print(f"● Top words: {top_str}")

if __name__ == "__main__":
    main()