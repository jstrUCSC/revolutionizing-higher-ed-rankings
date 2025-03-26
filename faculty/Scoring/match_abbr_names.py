import csv
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

INPUT_CSV = "author_paper_data.csv"
OUTPUT_CSV = "faculty_full_names.csv"

def fetch_dblp_authors_and_title(title, abbreviated_authors):
    """Fetch full author names and paper title from DBLP using the given paper title and verify with author names."""
    search_url = f"https://dblp.org/search?q={urllib.parse.quote(title)}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        print(f"Failed to fetch DBLP data for: {title}")
        return [], None

    print(f"Fetched DBLP data for: {title}")
    soup = BeautifulSoup(response.text, "html.parser")  # Use 'html.parser' for simplicity
    
    # Find all search results
    results = soup.find_all("cite", class_="data tts-content")

    for result in results:
        # Extract title (this is optional if you want to verify the exact paper)
        title_element = result.find("span", class_="title")
        if not title_element:
            continue
        fetched_title = title_element.get_text().strip()
        
        # Extract authors from the search result
        author_spans = result.find_all("span", itemprop="name")
        full_authors = [span.get_text().strip() for span in author_spans]

        # Check if at least one abbreviated author appears in the full author list
        for abbrev_author in abbreviated_authors:
            if any(abbrev_author.split()[-1].strip("'") in full_name for full_name in full_authors):
                return full_authors[:-2], fetched_title  # Return the correct author list and the paper title

    print(f"No exact match found for: {title}")
    return [], None

def process_csv(input_csv, output_csv):
    """Read the input CSV, fetch author names, match them, and save to an output CSV."""
    existing_entries = set()  # Use a set to avoid duplicates
    
    # Read existing output CSV to prevent duplicate entries
    try:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    existing_entries.add(row[0])
    except FileNotFoundError:
        pass
    print(f"Existing entries loaded: {existing_entries}")

    results = []  # Store results before writing

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            paper_title = row.get("Paper Title", "").strip()
            authors = row.get("Authors", "").strip()
            if "." in authors:
                author_list = [a.strip() for a in row.get("Authors", "").split(",") if a.strip()]
            else:
                for a in authors.split(","):
                    results.append([a.strip(), paper_title])
                continue

            print(f"Processing: {paper_title}")
            full_authors, fetched_title = fetch_dblp_authors_and_title(paper_title, author_list)
            
            if fetched_title:
                print(fetched_title)
                for name in full_authors:
                    if name not in existing_entries:
                        results.append([name, fetched_title])
                        existing_entries.add(name)

            time.sleep(1)  # To avoid hitting DBLP too frequently

    # Write to output CSV (overwrite to ensure clean output)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Full Name", "Paper Title"])  # Header
        writer.writerows(results)

    print("Processing complete. Results saved.")

def main():
    """Main function to execute the CSV processing."""
    process_csv(INPUT_CSV, OUTPUT_CSV)

if __name__ == "__main__":
    main()
