import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests, json
import csv
import re
import time
from crossref.restful import Journals, Works

university_scoring = {}

FILE_NAME = "survey-response/responses_edited_4_30.csv"
CSV_DEBUG = False
CSV_CAREFUL_SELECT = False


class Paper:
    def __init__(self, title, authors, fromCategory):
        self.title = title
        self.authors = authors
        self.fromCategory = fromCategory


"""
 ______   _______  _______    _______  ______    _______  _______ 
|      | |       ||       |  |       ||    _ |  |   _   ||  _    |
|  _    ||   _   ||       |  |    ___||   | ||  |  |_|  || |_|   |
| | |   ||  | |  ||       |  |   | __ |   |_||_ |       ||       |
| |_|   ||  |_|  ||      _|  |   ||  ||    __  ||       ||  _   | 
|       ||       ||     |_   |   |_| ||   |  | ||   _   || |_|   |
|______| |_______||_______|  |_______||___|  |_||__| |__||_______|
"""


def get_csv_data():
    df = pd.read_csv(FILE_NAME)
    df['Papers'] = df.apply(extract_papers, axis=1)
    papers_array = []
    for index, row in df.iterrows():
        for paper in row['Papers']:
            author_string = ', '.join(paper['Authors'])
            names = re.findall(r'[A-Z][a-z]*\s[A-Z][a-z]*', author_string)
            cleaned_names = []
            for name in names:
                # Remove numbers
                name = re.sub(r'\d+', '', name)
                # Remove acronyms
                name = re.sub(r'\b[A-Z]{2,}\b', '', name)
                # Remove leading/trailing whitespaces
                name = name.strip()
                if name:
                    cleaned_names.append(name)
            papers_array.append(Paper(paper['Title'], cleaned_names, paper['fromCategory']))
        print()

    for paper in papers_array:
        print(f"Title: {paper.title}")
        print(f"Authors: {paper.authors}")
        print(f"fromCategory: {paper.fromCategory}")
        print()
    return papers_array


def extract_papers(row):
    papers = []
    for i in range(1, 6):
        paper_info = row[f'Paper {i}']
        if isinstance(paper_info, str):
            paper_title = paper_info.split(';')[0]
            authors = paper_info.split(';')[1:]
            papers.append({'Title': paper_title, 'Authors': authors, 'fromCategory': row.iloc[2]})
    return papers


"""
 _______  _______  __   __    _______  _______  _______  ______    _______  __   __ 
|       ||       ||  | |  |  |       ||       ||   _   ||    _ |  |       ||  | |  |
|       ||  _____||  |_|  |  |  _____||    ___||  |_|  ||   | ||  |       ||  |_|  |
|       || |_____ |       |  | |_____ |   |___ |       ||   |_||_ |       ||       |
|      _||_____  ||       |  |_____  ||    ___||       ||    __  ||      _||       |
|     |_  _____| | |     |    _____| ||   |___ |   _   ||   |  | ||     |_ |   _   |
|_______||_______|  |___|    |_______||_______||__| |__||___|  |_||_______||__| |__|
"""


def csv_search_person(person):
    """
    Inputs: The full name of a person all in one string
    Returns: The name of the affiliation, if one was found, nothing otherwise.
    Comments: Interprets the string literally, looking for an exact match of the author name in the name field.
    Return rate: ~12%
    """
    # Get the first letter of the person's first name
    first_letter = person.split()[0][0].lower()

    # Path to the directory containing CSV files
    directory = os.path.join(os.getcwd(), "names")

    # Path to the CSV file
    csv_file_path = os.path.join(directory, f"csrankings-{first_letter}.csv")

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        if (CSV_DEBUG):
            print(f"CSV file for '{person}' not found.")
        return

    # Read the CSV file and search for the person's affiliation
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].strip() == person:
                return row[1].strip()  # Assuming the affiliation is in the second column
    # print(f"Affiliation not found for '{person}'.")
    return 0


def csv_search_person_v2(person):
    """
    Inputs: The full name of a person all in one string
    Returns: The name of the affiliation, if one was found, nothing otherwise.
    Comments: This uses some interpretive string matching, so the names are broken up and matched one-by-one. Periods are capable of extending out to the rest of a single word/name, but only that single word/name.
    Return rate: ~26%
    """
    # Split the full name into components (first name, middle name, last name)
    names = person.split()

    # Path to the directory containing CSV files
    directory = os.path.join(os.getcwd(), "names")

    # Initialize affiliation variable to None
    affiliation = None

    # Construct regular expressions for each name component
    name_patterns = [re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE) for name in names]

    # Path to the CSV file
    csv_file_path = os.path.join(directory, f"csrankings-{names[0][0].lower()}.csv")

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        if (CSV_DEBUG):
            print(f"CSV file for '{person}' not found.")
        return

    # Read the CSV file and search for the person's affiliation
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Check if all name components match using regular expressions
            if all(pattern.search(row[0]) for pattern in name_patterns):
                if (CSV_DEBUG):
                    print(f"Match found with {row[0].lower()}")
                affiliation = row[1].strip()  # Assuming the affiliation is in the second column
                break

    if affiliation:
        return affiliation
    else:
        if (CSV_DEBUG):
            print(f"Affiliation not found for '{person}'.")
        return 0


def csv_search(paper):


    if (CSV_DEBUG):
        print(f"\nPaper: {paper.title} | Authors: {paper.authors}")
    found = False
    for person in paper.authors:
        if (CSV_CAREFUL_SELECT):
            aff = csv_search_person(person)
        else:
            aff = csv_search_person_v2(person)
        if aff != 0:
            found = True
            num_authors = len(paper.authors)
            points_per_affiliation = 1 / num_authors
            print("CSV: Adding " + str(points_per_affiliation) + " points to " + str(aff))
            if aff in university_scoring:
                # If the university is already in the dictionary, increment the point value
                university_scoring[aff] += points_per_affiliation
            else:
                # If the university is not in the dictionary, add it with its point value
                university_scoring[aff] = points_per_affiliation
        else:
            found = False

    return found


"""
 ______   _______  ___     _______  _______  _______  ______    _______  __   __ 
|      | |       ||   |   |       ||       ||   _   ||    _ |  |       ||  | |  |
|  _    ||   _   ||   |   |  _____||    ___||  |_|  ||   | ||  |       ||  |_|  |
| | |   ||  | |  ||   |   | |_____ |   |___ |       ||   |_||_ |       ||       |
| |_|   ||  |_|  ||   |   |_____  ||    ___||       ||    __  ||      _||       |
|       ||       ||   |    _____| ||   |___ |   _   ||   |  | ||     |_ |   _   |
|______| |_______||___|   |_______||_______||__| |__||___|  |_||_______||__| |__|
"""


def find_doi(paper):
    works = Works()
    query = f"title:{paper.title}"
    for author in paper.authors:
        query += f" AND author:{author}"
    results = works.query(query)
    for result in results:
        # print(result.get('DOI'))
        return result.get('DOI')

    # return doi_list


def get_author_affiliation_doi(doi):
    works = Works()
    try:
        paper = works.doi(doi)
        authors = paper.get('author', [])
        print(authors)
        affiliations = []
        for author in authors:
            if 'affiliation' in author:
                affiliations.extend(author['affiliation'])
        return affiliations
    except Exception as e:
        print(f"DOI: Could not fetch data for {doi}")
        return []


def doi_search(paper):
    # Torin
    affiliations = get_author_affiliation_doi(find_doi(paper))
    if len(affiliations) == 0:
        return False
    print(affiliations)
    points_per_affiliation = 1 / len(affiliations)
    pattern = r'^([^,]+)'
    for university in affiliations:
        print("DOI: Adding " + str(points_per_affiliation) + " points to " + str(re.match(pattern, university['name']).group(1)))
        if re.match(pattern, university['name']).group(1) in university_scoring:
            # If the university is already in the dictionary, increment the point value
            university_scoring[re.match(pattern, university['name']).group(1)] += points_per_affiliation
        else:
            # If the university is not in the dictionary, add it with its point value
            university_scoring[re.match(pattern, university['name']).group(1)] = points_per_affiliation
    return True


"""
 _______  _______  __   __    _______  _______  _______  ______    _______  __   __ 
|   _   ||       ||  |_|  |  |       ||       ||   _   ||    _ |  |       ||  | |  |
|  |_|  ||       ||       |  |  _____||    ___||  |_|  ||   | ||  |       ||  |_|  |
|       ||       ||       |  | |_____ |   |___ |       ||   |_||_ |       ||       |
|       ||      _||       |  |_____  ||    ___||       ||    __  ||      _||       |
|   _   ||     |_ | ||_|| |   _____| ||   |___ |   _   ||   |  | ||     |_ |   _   |
|__| |__||_______||_|   |_|  |_______||_______||__| |__||___|  |_||_______||__| |__|
"""
"""Searches for the affiliations found within a paper in ACM HTML"""


def get_paper_info(paper_url):
    affiliations_list = []
    affiliations_double_list = []
    # Make an HTTP GET request to the ACM Digital Library
    response = requests.get(paper_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the response
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the HTML elements containing the authors and their affiliations
        for i in soup.find_all('p'):
            text = i.get_text(strip=True)
            if text.startswith('This alert'):
                break
            else:
                affiliations_double_list.append(text)
        for index, sublist in enumerate(affiliations_double_list):
            if index % 2 == 0:
                affiliations_list.append(sublist)

        # Return the authors and their affiliations
        return affiliations_list
    else:
        # If the request was not successful, print an error message
        print("Failed to retrieve paper information. Status code:", response.status_code)
        return None


"""Searches for the URL of a Paper and returns a URL if successful search in ACM Library"""


def search_paper(title):
    # Make a search request to the ACM Digital Library
    search_url = f"https://dl.acm.org/action/doSearch?AllField={title}"
    response = requests.get(search_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the response
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the first search result link starting with "/doi"
        result_link = soup.find('a', href=lambda href: href and href.startswith("/doi"))

        # Extract the link URL
        if result_link:
            paper_url = "https://dl.acm.org" + result_link['href']
            return paper_url
        else:
            print("No search results found.")
            return None
    else:
        # If the request was not successful, print an error message
        print("Failed to perform search. Status code:", response.status_code)
        return None


"""Returns a list of authors as well as the number of authors in ACM HTML"""


def get_author_data(paper_url):
    # Make an HTTP GET request to the ACM Digital Library
    response = requests.get(paper_url)
    author_list = []
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the response
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the HTML elements containing the authors' information
        author_data_sections = soup.find_all('div', class_='author-data')
        for i in author_data_sections:
            author_name = i.get_text(strip=True)
            if author_name:
                author_list.append(author_name)

        return author_list, len(author_list)


def acm_search(paper):
    # Aaron
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/58.0.3029.110 Safari/537.3'
    }
    session.headers.update(headers)
    found = False
    paper_url = search_paper(paper.title)
    # paper_url = 'https://dl.acm.org/doi/10.1145/3434393'
    if paper_url:
        print("ACM: Found paper URL:", paper_url)
        time.sleep(1)
        response = session.get(paper_url)
        if response.status_code == 200:
            # affiliations = get_paper_info(paper_url)
            affiliations = None
            try:
                affiliations = get_paper_info(response.text)
            except Exception as e:
                print("ACM: Failed to fetch paper")

            if affiliations:
                #print("Affiliations:", affiliations)
                author_data = get_author_data(response.text)
                # author_data = get_author_data(paper_url)
                # affiliations = get_paper_info(paper_url)
                if author_data[1] == len(affiliations):
                    found = True
                    num_authors = author_data[1]
                    points_per_affiliation = 1 / num_authors
                    for university in affiliations:
                        print("ACM: Adding " + str(points_per_affiliation) + " points to " + str(university))
                        if university in university_scoring:
                            # If the university is already in the dictionary, increment the point value
                            university_scoring[university] += points_per_affiliation
                        else:
                            # If the university is not in the dictionary, add it with its point value
                            university_scoring[university] = points_per_affiliation
        else:
            print("Failed to fetch paper:", response.status_code)

    sorted_university_scoring = sorted(university_scoring.items(), key=lambda x: x[1], reverse=True)
    university_scores = dict(sorted_university_scoring)
    return found


"""
 _______  __   __  ___      ___        _______  _______  _______  ______    _______  __   __ 
|       ||  | |  ||   |    |   |      |       ||       ||   _   ||    _ |  |       ||  | |  |
|    ___||  | |  ||   |    |   |      |  _____||    ___||  |_|  ||   | ||  |       ||  |_|  |
|   |___ |  |_|  ||   |    |   |      | |_____ |   |___ |       ||   |_||_ |       ||       |
|    ___||       ||   |___ |   |___   |_____  ||    ___||       ||    __  ||      _||       |
|   |    |       ||       ||       |   _____| ||   |___ |   _   ||   |  | ||     |_ |   _   |
|___|    |_______||_______||_______|  |_______||_______||__| |__||___|  |_||_______||__| |__|
"""

def save_dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        writer.writeheader()
        writer.writerow(dictionary)
def run_full_search():
    """
    1. Search Through CS Rankings CSV files
    2. DOI search
    3. ACM Search
    4. LLM Search
    """
    papers = get_csv_data()
    num_authors = 0
    num_successes = 0
    for paper in papers:
        num_authors += len(paper.authors)
        if not doi_search(paper):
            if not acm_search(paper):
                if not csv_search(paper):
                    print(f"FULL: Could not find affil for {paper.title}")
                else:
                    num_successes += 1
            else:
                num_successes += 1
        else:
            num_successes += 1

    print(f"FULL: Found afffil for {num_successes} papers out of {len(papers)}")
    save_dict_to_csv(university_scoring, 'u_scores.csv')
    # print(f"DOI Search: Found {len(doi_search_results)} out {num_authors} affiliations")

    # print(acm_search(papers))
    return


if __name__ == "__main__":
    # get_csv_data()
    run_full_search()
