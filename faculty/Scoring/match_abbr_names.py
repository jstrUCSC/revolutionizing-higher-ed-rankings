import json
import csv
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

INPUT_JSON = "../../llm/extracted_references.json"
OUTPUT_CSV = "faculty_full_names.csv"

def fetch_dblp_authors_and_title(title, abbreviated_authors):
    """Fetch disambiguated full author names from DBLP using the given paper title."""
    search_url = f"https://dblp.org/search?q={urllib.parse.quote(title)}"
    response = requests.get(search_url)

    if response.status_code != 200:
        print(f"Failed to fetch DBLP data for: {title}")
        return []

    print(f"Fetched DBLP data for: {title}")
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("cite", class_="data tts-content")

    for result in results:
        author_spans = result.find_all("span", itemprop="name")
        full_authors = [
            span.get("title", span.get_text().strip()).strip()
            for span in author_spans[:-2] # Exclude paper title and conference
        ]

        print(f"Full authors found: {full_authors}")

        for abbrev_author in abbreviated_authors:
            if any(abbrev_author.split()[-1].strip("'") in full_name for full_name in full_authors):
                return full_authors

    print(f"No exact match found for: {title}")
    return []



def process_json(input_json, output_csv):
    existing_entries = set()
    
    # Read existing output to prevent duplicates
    try:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    existing_entries.add(row[0])
    except FileNotFoundError:
        pass
    print(f"Existing entries loaded: {existing_entries}")

    results = []

    with open(input_json, "r", encoding="utf-8") as f:
        papers = json.load(f)

        for paper in papers:
            title = paper.get("title", "").strip()
            authors = paper.get("authors", [])
            original_authors = ["Pierre-Étienne H. Fiquet", "Eero P. Simoncelli"]

            if not title or not authors or not original_authors:
                continue

            # Check if abbreviated (presence of ".")
            if any("." in a for a in authors):
                print(f"Processing: {title}")
                full_names = fetch_dblp_authors_and_title(title, authors)
                if full_names:
                    for name in full_names:
                        if name not in existing_entries:
                            results.append([name, 1 / (len(original_authors) * len(authors))])
                            existing_entries.add(name)
                else:
                    print(f"Could not find full names for: {title}")
                time.sleep(1)
            else:
                for a in authors:
                    name = a.strip()
                    if name and name not in existing_entries:
                        results.append([name, 1 / (len(original_authors) * len(authors))])
                        existing_entries.add(name)

    # Write results
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Full Name", "Normalized Score"])
        writer.writerows(results)

    print("Processing complete. Results saved.")

def main():
    process_json(INPUT_JSON, OUTPUT_CSV)

if __name__ == "__main__":
    main()
