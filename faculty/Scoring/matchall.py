import json
import csv
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import os
import re
import glob
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
from difflib import SequenceMatcher

JSON_DIR = "/nfs/stak/users/wangl9/hpc-share/revolutionizing-higher-ed-rankings/AT"
GLOB_PATTERN = os.path.join(JSON_DIR, "ACL20??_re.json")  

author_cache = {}

def normalize_name_for_matching(name):
    normalized = name.replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized.lower()

def are_names_similar(name1, name2, threshold=0.85):
    norm1 = normalize_name_for_matching(name1)
    norm2 = normalize_name_for_matching(name2)
    
    if norm1 == norm2:
        return True
    
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity >= threshold

def fetch_dblp_canonical_name(query_name):
    
    if query_name in author_cache:
        return author_cache[query_name]
    
    search_url = f"https://dblp.org/search/author?q={urllib.parse.quote(query_name)}"
    
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=10,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.get(search_url)
    except requests.exceptions.RequestException as e:
        print(f"Request failed for author '{query_name}': {e}")
        author_cache[query_name] = query_name 
        return query_name
    
    if response.status_code != 200:
        print(f"Failed to fetch DBLP data for: {query_name} (status {response.status_code})")
        author_cache[query_name] = query_name
        return query_name
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    results = soup.find_all("li", class_="entry")
    
    for result in results:
        author_elem = result.find("span", itemprop="name")
        if not author_elem:
            continue
            
        canonical_name = author_elem.get("title", author_elem.get_text()).strip()
        
        if are_names_similar(query_name, canonical_name):
            print(f"Mapped '{query_name}' -> '{canonical_name}'")
            author_cache[query_name] = canonical_name
            return canonical_name
    
    print(f"No canonical name found for: {query_name}, using original")
    author_cache[query_name] = query_name
    return query_name

def fetch_dblp_authors_and_title(title, abbreviated_authors):
    search_url = f"https://dblp.org/search?q={urllib.parse.quote(title)}"

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=10,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(search_url)
    except requests.exceptions.RequestException as e:
        print(f"Request failed for title '{title}': {e}")
        return []

    if response.status_code != 200:
        print(f"Failed to fetch DBLP data for: {title} (status {response.status_code})")
        return []

    print(f"Fetched DBLP data for: {title}")
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("cite", class_="data tts-content")

    for result in results:
        author_spans = result.find_all("span", itemprop="name")
        full_authors = [
            span.get("title", span.get_text().strip()).strip()
            for span in author_spans[:-2]
        ]

        print(f"Full authors found: {full_authors}")

        for abbrev_author in abbreviated_authors:
            # Add null/empty checking here
            if abbrev_author is None or not abbrev_author.strip():
                continue
            
            # Split and check if the list is not empty before accessing [-1]
            author_parts = abbrev_author.split()
            if not author_parts:
                continue
                
            last_name = author_parts[-1].strip("'")
            if any(last_name in full_name for full_name in full_authors):
                return full_authors

    print(f"No exact match found for: {title}")
    return []

def consolidate_similar_names(scores):
    consolidated = {}
    processed = set()
    
    names = list(scores.keys())
    
    for i, name1 in enumerate(names):
        if name1 in processed:
            continue
            
        similar_names = [name1]
        total_score = scores[name1]
        
        for j, name2 in enumerate(names[i+1:], i+1):
            if name2 in processed:
                continue
            if are_names_similar(name1, name2):
                similar_names.append(name2)
                total_score += scores[name2]
                processed.add(name2)
        
        canonical = max(similar_names, key=lambda x: (len(x), '-' in x))
        consolidated[canonical] = total_score
        processed.add(name1)
    
    return consolidated

def read_existing_scores(output_csv):
    scores = {}
    if not os.path.exists(output_csv):
        return scores

    try:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip()
                    try:
                        score = float(row[1])
                    except ValueError:
                        score = 0.0
                    if name:
                        scores[name] = scores.get(name, 0.0) + score
    except FileNotFoundError:
        pass

    print(f"Loaded {len(scores)} existing authors with scores.")
    return scores

def write_scores(output_csv, scores, sort_by_score=False):
    """Rewrite the entire CSV from scores dict."""
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Full Name", "Normalized Score"])
        if sort_by_score:
            items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for name, score in items:
                writer.writerow([name, score])
        else:
            for name in sorted(scores.keys()):
                writer.writerow([name, scores[name]])

def process_single_json(input_json, output_csv, sort_by_score=False, normalize_all=False):
    scores = read_existing_scores(output_csv)

    with open(input_json, "r", encoding="utf-8") as f:
        papers = json.load(f)
        total_papers = len(papers)

        for idx, paper in enumerate(papers, 1):
            print(f"\n[Progress] ({os.path.basename(input_json)}) {idx}/{total_papers}")

            if paper is None:
                print(f"Skipping None entry at index {idx}")
                continue
            if not isinstance(paper, dict):
                print(f"Skipping non-dictionary entry at index {idx}: {type(paper)}")
                continue

            title = paper.get("title", "")
            if title is None:
                title = ""
            if isinstance(title, list):
                title = str(title[0]) if len(title) > 0 else ""
            title = str(title).strip()

            authors = paper.get("authors", [])
            if authors is None:
                authors = []

            if not title or not authors:
                print(f"Skipping entry {idx} due to missing title or authors")
                continue

            try:
                increment = 1.0 / float(len(authors))
            except ZeroDivisionError:
                print(f"Skipping entry {idx} due to zero authors")
                continue

            if any("." in a for a in authors if a is not None):
                print(f"Processing (abbreviated): {title}")
                full_names = fetch_dblp_authors_and_title(title, authors)
                if full_names:
                    for name in full_names:
                        clean = name.strip()
                        if clean:
                            scores[clean] = scores.get(clean, 0.0) + increment
                else:
                    print(f"Could not find full names for: {title}")
                time.sleep(random.uniform(15, 30))
            else:
                for a in authors:
                    if a is None:
                        continue
                    clean = a.strip()
                    if not clean:
                        continue
                    
                    if normalize_all:
                        canonical = fetch_dblp_canonical_name(clean)
                        scores[canonical] = scores.get(canonical, 0.0) + increment
                        if canonical != clean:
                            time.sleep(random.uniform(5, 10))  
                    else:
                        scores[clean] = scores.get(clean, 0.0) + increment

    print("\nConsolidating similar author names...")
    scores = consolidate_similar_names(scores)
    
    write_scores(output_csv, scores, sort_by_score=sort_by_score)
    print(f"Done: {output_csv}")

def main():
    files = sorted(glob.glob(GLOB_PATTERN))
    if not files:
        print(f"No files matched: {GLOB_PATTERN}")
        return

    print("Found JSON files:")
    for p in files:
        print(" -", p)

    for json_path in files:
        base = os.path.basename(json_path)
        m = re.search(r"(ACL20\d{2})_re\.json$", base)
        if not m:
            print(f"Skip (cannot parse pattern): {base}")
            continue
        prefix = m.group(1)

        output_csv = os.path.join(os.path.dirname(json_path), f"{prefix}_fulln.csv")
        print(f"\n=== Processing {base} -> {os.path.basename(output_csv)} ===")
        
        process_single_json(json_path, output_csv, sort_by_score=False, normalize_all=True)

if __name__ == "__main__":
    main()