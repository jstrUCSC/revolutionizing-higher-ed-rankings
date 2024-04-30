import sys
import pandas as pd
from bs4 import BeautifulSoup
import requests, json
import csv
import re

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



def run_full_search():
    """
    1. Search Through CS Rankings CSV files
    2. DOI search
    3. ACM Search
    4. LLM Search
    """
    papers = get_csv_data()
    return


def csv_search(papers):
    # Elliot
    return


def doi_search(papers):
    # Torin
    return


def acm_search(papers):
    # Aaron
    return


def llm_search(papers):
    return

if __name__ == "__main__":
    get_csv_data()
