import csv
import os
import urllib.request
import time
# pip install beautifulsoup4
# an html5 reader
from bs4 import BeautifulSoup 

""" DO NOT DOWNLOAD ALL PUBLICATIONS WHEN TESTING

    - It will take over 10 hours
    - Files are not checked if they exist
    
"""
# Downloads all publications from all conferences within the conferences folder.
# * Each publication is downloaded into a folder with the name of the conference
#   CSV it is a from, without "_papers.csv"
# NOTE: Directory management only tested on a Windows machine
def download_all():
    conferences = os.listdir(os.getcwd() + "/Conferences")
    for file_name in conferences:
       download_conference(file_name)
    return

# Downloads all publications from a specific conference
# input: the CSV file name (include extension)
def download_conference(file_name):
    with open(os.getcwd() + "/Conferences/" + file_name, encoding="utf8") as conf_csv:
        url_reader = csv.reader(conf_csv, delimiter=',', quotechar='\"')
        row = next(url_reader)
        while True:
            try:
                row = next(url_reader)
                url = row[4]
                paper = download_paper(url, file_name)
                if not paper == -4:
                    time.sleep(1)
            except StopIteration:
                break
    return
    

# Downloads a paper from the url parameter.
# Assumes that the link provided is not a direct link to the pdf (which is
# the case with the current publication CSVs).
# arxiv.org and doi.org not supported.
def download_paper(url, conference):
    if not ".html" in url:
        print("Invalid URL (DOI or ARXIV?)")
        return -4
    success = 0
    domains = ["http://papers.nips.cc", "http://proceedings.mlr.press",
               "https://proceedings.mlr.press", "https://proceedings.neurips.cc",
               "https://openaccess.thecvf.com", "https://doi.org", "arxiv.org"]
    if domains[5] in url or domains[6] in url:
               return # Cannot handle DOI or Arxiv links
    html = urllib.request.urlopen(url).read().decode("utf8")
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all('a')
    conference_folder = "Publications/" + conference[:-11] + "/" # Remove "_papers.csv" suffix
    download_link = ""
    name = ""
    #print(url)
    # Check each link for a valid PDF download
    for link in links:
        if domains[0] in url:   # papers.nips.cc and proceedings.neurips.cc require extra care
            if link.string == "Paper":
                download_link = domains[0] + "/" + link.get("href")
                name = soup.find_all("h4")[0].string
                
        elif domains[1] in url or domains[2] in url: # proceedings.mlr.press
            if link.string == "Download PDF":
                download_link = link.get("href")
                name = soup.find_all("h1")[0].string

        elif domains[3] in url:
            if link.string == "Paper":
                download_link = domains[3] + "/" + link.get("href")
                name = soup.find_all("h4")[0].string

        elif domains[4] in url: # openaccess.thecvf.com
            if link.string == "pdf":
                if not "html/w" in url:
                    download_link = domains[4] + link.get("href")[5:] # Remove ../.. beginning
                else:    
                    download_link = domains[4] + link.get("href")[8:] # Remove ../../.. beginning
                if soup.find(id="papertitle").string[0] == '\n':
                    name = soup.find(id="papertitle").string[1:]
                else:
                    name = soup.find(id="papertitle").string

    # Set the name of the PDF to the name of the publication (with some format changes)
    # * Some extra "replace" cases may be necessary for special characters later, these were
    #   the only ones I ran into
    name = name.replace(" ", "_").replace(",", "").replace(":","").replace("$\\alpha$", "alpha")
    local_path = conference_folder + name + ".pdf"
    print(download_link)
    stored_loc = urllib.request.urlretrieve(download_link, local_path)
    if(stored_loc):
        print("Success")
        return 1
    else:
        print("Could not download")
        return 0
   
def main():
    publication_path = os.getcwd() + "/Publications"
    conferences_path = os.listdir(os.getcwd() + "/Conferences")
                    
    if not os.path.exists(publication_path):
        os.makedirs(publication_path)
    
    for file_name in conferences_path:
        publication_path = os.getcwd() + "/Publications/" + file_name[:-11] # Remove "_papers.csv" suffix
        if not os.path.exists(publication_path):
            os.makedirs(publication_path)
                             
    download_conference("NeurIPS_2023_papers.csv")
    #download_all()

if __name__ == "__main__":
    main()
