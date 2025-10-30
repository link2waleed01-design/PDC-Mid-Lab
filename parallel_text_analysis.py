#!/usr/bin/env python3
"""
parallel_text_analysis.py

Split reviews.csv into equal chunks per worker, count words in parallel,
merge results, and report timing for 1, 2, 4, and 8 workers in a table.
"""
import csv
import re
import time
from collections import Counter
from multiprocessing import Pool

INPUT_FILE = "reviews.csv"
MAX_ROWS = 20000
WORKER_LIST = (1, 2, 4, 8)

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
    """Count tokens for a list of text strings."""
    c = Counter()
    for t in texts:
        c.update(clean_and_tokenize(t))
    return c

def chunkify(seq, n):
    """Split seq into n nearly-equal chunks."""
    length = len(seq)
    if n <= 0:
        return []
    base = length // n
    rem = length % n
    chunks = []
    start = 0
    for i in range(n):
        extra = 1 if i < rem else 0
        end = start + base + extra
        chunks.append(seq[start:end])
        start = end
    return chunks

def load_texts(path, max_rows):
    texts = []
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
            if text_idx < len(row):
                texts.append(row[text_idx])
            else:
                texts.append("")
    return texts

def run_with_pool(chunks, workers):
    with Pool(processes=workers) as p:
        local_counters = p.map(worker_count, chunks)
    total = Counter()
    for lc in local_counters:
        total.update(lc)
    return total

def measure_time(texts, workers):
    # For workers == 1, run sequentially (no Pool) to get baseline timing without pool overhead.
    if workers == 1:
        start = time.perf_counter()
        total = worker_count(texts)
        elapsed = time.perf_counter() - start
        return total, elapsed
    # For >1 workers, chunk and use multiprocessing pool
    chunks = chunkify(texts, workers)
    start = time.perf_counter()
    total = run_with_pool(chunks, workers)
    elapsed = time.perf_counter() - start
    return total, elapsed

def print_markdown_table(rows):
    # rows: list of dicts with keys workers, time, speedup, efficiency
    print("| Workers | Time (s) | Speedup | Efficiency (%) |")
    print("| ------- | -------- | ------- | --------------:|")
    for r in rows:
        print(f"| {r['workers']:>1d}      | {r['time']:6.1f}   | {r['speedup']:5.2f}x | {r['efficiency']:6.1f} |")

if __name__ == "__main__":
    texts = load_texts(INPUT_FILE, MAX_ROWS)
    processed = len(texts)
    if processed == 0:
        print("No reviews loaded.")
    else:
        results = []
        baseline_time = None
        # measure for each worker count (including 1 for baseline)
        for w in WORKER_LIST:
            total_counter, elapsed = measure_time(texts, w)
            if w == 1:
                baseline_time = elapsed
                speedup = 1.0
                efficiency = 100.0
            else:
                speedup = baseline_time / elapsed if elapsed > 0 else float('inf')
                efficiency = (speedup / w) * 100.0
            results.append({
                "workers": w,
                "time": elapsed,
                "speedup": speedup,
                "efficiency": efficiency
            })
        # print markdown-style table similar to the example
        print_markdown_table(results)