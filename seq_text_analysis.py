#!/usr/bin/env python3
"""
seq_text_analysis.py

Load the first 20,000 rows from reviews.csv, clean text, remove stopwords,
count word frequencies, write the top 20 most frequent words to seq_output.csv,
and print total processing time.
"""
import csv
import re
import time
import sys
from collections import Counter

INPUT_FILE = "reviews.csv"
OUTPUT_FILE = "seq_output.csv"
MAX_ROWS = 20000
TOP_N = 20

TOKEN_RE = re.compile(r"[a-z']+")  # keep letters and apostrophes (e.g., don't)

# Basic English stopwords (baseline)
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
    # fallback: choose first column after common id columns
    if len(lower) == 1:
        return 0
    # prefer a non-id column if possible
    for i, h in enumerate(lower):
        if not any(token in h for token in ("id", "rating", "score", "date")):
            return i
    return 0

def clean_and_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    # remove stopwords and single-character non-informative tokens (optional)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def process_file(input_path, max_rows):
    counter = Counter()
    processed = 0
    try:
        with open(input_path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return counter, processed  # empty file
            text_idx = detect_text_column(header)
            for row in reader:
                if processed >= max_rows:
                    break
                processed += 1
                if text_idx >= len(row):
                    continue
                tokens = clean_and_tokenize(row[text_idx])
                if tokens:
                    counter.update(tokens)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return Counter(), 0
    return counter, processed

def write_top_n(output_path, most_common):
    with open(output_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["word", "count"])
        for word, count in most_common:
            writer.writerow([word, str(count)])

def main():
    start = time.perf_counter()
    counts, processed = process_file(INPUT_FILE, MAX_ROWS)
    top = counts.most_common(TOP_N)
    write_top_n(OUTPUT_FILE, top)
    elapsed = time.perf_counter() - start

    # Print results in the requested format
    print(f"● Processed {processed:,} reviews in {elapsed:.1f} seconds.")
    top_words_str = ", ".join(f"{w}({c})" for w, c in top) + ","
    print(f"● Top words: {top_words_str}")

if __name__ == "__main__":
    main()