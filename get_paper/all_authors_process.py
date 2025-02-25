import pandas as pd
import glob

def extract_authors_universities():
    # Combine all paper CSVs that we currently have, we can add more later. 
    papers_files = glob.glob("./Conferences/CVPR_*.csv") + glob.glob("./Conferences/ICML_*.csv") + glob.glob("./Conferences/NeurIPS_*.csv") + glob.glob("./Conferences/IJCAI_*.csv") + glob.glob("./Conferences/KDD_*.csv")
    papers_list = [pd.read_csv(file) for file in papers_files]
    papers = pd.concat(papers_list, ignore_index=True)

    # Combine all csrankings-(a-z).csv files from CSrankings git repo.
    csrankings_files = glob.glob("../CSrankings/csrankings-*.csv")
    csrankings_list = [pd.read_csv(file) for file in csrankings_files]
    csrankings = pd.concat(csrankings_list, ignore_index=True)

    # Create a dictionary mapping author names to their universities.
    author_university_map = dict(zip(csrankings['name'], csrankings['affiliation']))

    # Initialize a list to store authors and their universities, allowing duplicates.
    author_university_list = []

    # Iterate through the combined papers data.
    for index, row in papers.iterrows():
        if pd.isna(row['Authors']):  # Skip rows with missing authors.
            continue

        authors = row['Authors'].split(",")  
        for author in authors:
            author = author.strip()  
            university = author_university_map.get(author)  
            if university:  # Add authors with found universities.
                author_university_list.append((author, university))

    # Convert the results into a DataFrame, allowing duplicates
    result_df = pd.DataFrame(author_university_list, columns=["Author", "University"])

    # Print the result to output csv file
    result_df.to_csv("author_universities_output.csv", index=False)

    return result_df

# Run the function and generate output
author_university_df = extract_authors_universities()
