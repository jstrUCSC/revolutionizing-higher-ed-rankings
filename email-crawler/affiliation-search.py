import sys
import pandas as pd
from bs4 import BeautifulSoup
import requests, json
import csv
import re
import time

university_scoring = {}

FILE_NAME = "survey-response/responses_edited_4_30.csv"

class Paper:
    def __init__(self, title, authors):
        self.title = title
        self.authors = authors
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
            papers_array.append(Paper(paper['Title'], cleaned_names))
        print()

    for paper in papers_array:
        print(f"Title: {paper.title}")
        print(f"Authors: {paper.authors}")
    return papers_array


def extract_papers(row):
    papers = []
    for i in range(1, 6):
        paper_info = row[f'Paper {i}']
        if isinstance(paper_info, str):
            paper_title = paper_info.split(';')[0]
            authors = paper_info.split(';')[1:]
            papers.append({'Title': paper_title, 'Authors': authors})
    return papers

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

def run_full_search():
    """
    1. Search Through CS Rankings CSV files
    2. DOI search
    3. ACM Search
    4. LLM Search
    """
    papers = get_csv_data()
    print(acm_search(papers))
    return


def csv_search(papers):
    # Elliot
    return


def doi_search(papers):
    # Torin
    return


def acm_search(papers):
    # Aaron
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/58.0.3029.110 Safari/537.3'
    }
    session.headers.update(headers)
    for paper in papers:
        paper_url = search_paper(paper.title)
        #paper_url = 'https://dl.acm.org/doi/10.1145/3434393'
        if paper_url:
            print("Found paper URL:", paper_url)
            time.sleep(1)
            response = session.get(paper_url)
            if response.status_code == 200:
                # affiliations = get_paper_info(paper_url)
                affiliations = get_paper_info(response.text)
                if affiliations:
                    print("Affiliations:", affiliations)
                    author_data = get_author_data(response.text)
                    # author_data = get_author_data(paper_url)
                    # affiliations = get_paper_info(paper_url)
                    if author_data[1] == len(affiliations):
                        num_authors = author_data[1]
                        points_per_affiliation = 1 / num_authors
                        for university in affiliations:
                            print("Adding " + str(points_per_affiliation) + " points to " + str(university))
                            if university in university_scoring:
                                # If the university is already in the dictionary, increment the point value
                                university_scoring[university] += points_per_affiliation
                            else:
                                # If the university is not in the dictionary, add it with its point value
                                university_scoring[university] = points_per_affiliation
            else:
                print("Failed to fetch paper:", response.status_code)

    sorted_university_scoring = sorted(university_scoring.items(), key=lambda x:x[1], reverse=True)
    university_scores = dict(sorted_university_scoring)
    return university_scores


def llm_search(papers):
    return


if __name__ == "__main__":
    # get_csv_data()
    run_full_search()
