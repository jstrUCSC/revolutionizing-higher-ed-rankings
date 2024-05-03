import requests
from bs4 import BeautifulSoup


university_scoring = {}


"""Searches for the affiliations found within a paper"""
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


"""Searches for the URL of a Paper and returns a URL if successful search"""
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


"""Returns a list of authors as well as the number of authors"""
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


title = "Theoretically Efficient Parallel Graph Algorithms Can Be Fast and Scalable"
paper_url = search_paper(title)
if paper_url:
    print("Found paper URL:", paper_url)
    affiliations = get_paper_info(paper_url)
    if affiliations:
        print("Affiliations:", affiliations)
        author_data = get_author_data(paper_url)
        affiliations = get_paper_info(paper_url)
        if author_data[1] == len(affiliations):
            num_authors = author_data[1]
            print('Same Length of Authors and Affiliations')
            points_per_affiliation = 1/num_authors
            for university in affiliations:
                if university in university_scoring:
                    # If the university is already in the dictionary, increment the point value
                    university_scoring[university] += points_per_affiliation
                else:
                    # If the university is not in the dictionary, add it with its point value
                    university_scoring[university] = points_per_affiliation
else:
    print("Failed to find paper.")

print(university_scoring)
