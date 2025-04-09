import pandas as pd
import os

CONFERENCES_CATEGORIES = {
    'AAAI': 'Artificial Intelligence & Machine Learning',
    'IJCAI': 'Artificial Intelligence & Machine Learning',
    'NeurIPS': 'Artificial Intelligence & Machine Learning',
    'ICML': 'Artificial Intelligence & Machine Learning',
    'KDD': 'Data Science & Data Mining',
    'ICDM': 'Data Science & Data Mining',
    'SDM': 'Data Science & Data Mining',
    'CIKM': 'Data Science & Data Mining',
    'WWW': 'Data Science & Data Mining',
    'CVPR': 'Computer Vision & Image Processing',
    'ICCV': 'Computer Vision & Image Processing',
    'ECCV': 'Computer Vision & Image Processing',
    'ACL': 'Natural Language Processing',
    'EMNLP': 'Natural Language Processing',
    'NAACL': 'Natural Language Processing',
    'COLING': 'Natural Language Processing',
    'SOSP': 'Systems & Networking',
    'OSDI': 'Systems & Networking',
    'SIGCOMM': 'Systems & Networking',
    'NSDI': 'Systems & Networking',
    'PODC': 'Systems & Networking',
    'DISC': 'Systems & Networking',
    'FAST': 'Systems & Networking',
    'SIGMOD': 'Databases',
    'VLDB': 'Databases',
    'ICDE': 'Databases',
    'EDBT': 'Databases',
    'CCS': 'Security & Privacy',
    'USENIX Security': 'Security & Privacy',
    'NDSS': 'Security & Privacy',
    'S&P (Oakland)': 'Security & Privacy',
    'CHI': 'Human Computer Interaction',
    'UIST': 'Human Computer Interaction',
    'STOC': 'Theoretical Computer Science',
    'FOCS': 'Theoretical Computer Science',
    'ICALP': 'Theoretical Computer Science',
    'SODA': 'Theoretical Computer Science',
    'ICSE': 'Software Engineering',
    'FSE': 'Software Engineering',
    'ASE': 'Software Engineering',
    'ISSTA': 'Software Engineering',
    'SIGGRAPH': 'Computer Graphics & Virtual Reality',
    'Eurographics': 'Computer Graphics & Virtual Reality',
    'IEEE VR': 'Computer Graphics & Virtual Reality',
    'QIP': 'Quantum Computing',
    'TQC': 'Quantum Computing',
    'CoRL': 'Interdisciplinary Fields',
    'AMIA': 'Interdisciplinary Fields',
    'L@S': 'Interdisciplinary Fields',
}


def find_paper_titles(faculty_file, author_file, paper_file):
    """
    Finds paper titles for faculty members by cross-referencing with author_paper_data.csv.

    Args:
        faculty_file (str): Path to the faculty_full_names.csv file.
        author_file (str): Path to the author_universities_output.csv file.
        paper_file (str): Path to the author_paper_data.csv file.

    Returns:
        pd.DataFrame: DataFrame with faculty, university, and paper titles.
    """
    # Load the faculty_full_names.csv file
    faculty_data = pd.read_csv(faculty_file)

    # Load the author_universities_output.csv file
    author_data = pd.read_csv(author_file)

    # Load the author_paper_data.csv file
    paper_data = pd.read_csv(paper_file)

    # Create a dictionary to map authors to their universities
    author_to_university = author_data.drop_duplicates(subset=["Author"]).set_index("Author")["University"].to_dict()

    # Add university information for the "Reference Name" column in the faculty_data DataFrame
    faculty_data["Reference University"] = faculty_data["Reference Name"].map(author_to_university)

    # Drop references without university ties
    faculty_data = faculty_data.dropna(subset=["Reference University"])

    # Initialize a list to store paper titles
    paper_titles = []

    # Iterate through each reference name in the faculty data
    for reference_name in faculty_data["Reference Name"]:
        # Find rows in paper_data where the reference name appears in the "Authors" column
        matching_papers = paper_data[paper_data["Authors"].str.contains(reference_name, na=False)]
        
        # Collect the paper titles
        titles = matching_papers["Paper Title"].tolist()
        paper_titles.append(", ".join(titles) if titles else None)

    # Add the paper titles to the faculty_data DataFrame
    faculty_data["Paper Titles"] = paper_titles

    return faculty_data

def check_paper_in_conferences_with_categories(faculty_file, author_file, paper_file, conference_folder, conference_categories):
    """
    Checks if paper titles are present in the conference CSV files and assigns categories.

    Args:
        faculty_file (str): Path to the faculty_full_names.csv file.
        author_file (str): Path to the author_universities_output.csv file.
        paper_file (str): Path to the author_paper_data.csv file.
        conference_folder (str): Path to the folder containing conference CSVs.
        conference_categories (dict): Dictionary mapping conferences to their categories.

    Returns:
        pd.DataFrame: DataFrame with faculty, university, paper titles, matched conferences, and categories.
    """
    # Find paper titles and filter faculty with university ties
    faculty_data = find_paper_titles(faculty_file, author_file, paper_file)

    # Initialize lists to store matched conferences and categories
    matched_conferences = []
    matched_categories = []

    # Loop through each paper in the faculty data
    for paper_titles in faculty_data["Paper Titles"]:
        conferences = []
        categories = []

        if paper_titles:  # Check if there are paper titles
            # Loop through each file in the conference folder
            for file_name in os.listdir(conference_folder):
                if file_name.endswith(".csv"):
                    try:
                        # Load the conference CSV
                        conference_data = pd.read_csv(
                            os.path.join(conference_folder, file_name),
                            on_bad_lines="skip"  # Skip problematic lines
                        )
                        
                        # Normalize column names
                        conference_data.columns = conference_data.columns.str.strip()

                        # Check if 'Title' exists in the columns
                        if 'Title' not in conference_data.columns:
                            print(f"'Title' column not found in {file_name}. Skipping this file.")
                            continue

                        # Check if any paper title matches the "Title" column in the conference CSV
                        for paper_title in paper_titles.split(", "):
                            if paper_title in conference_data["Title"].values:
                                conference_key = file_name.split("_")[0]  # Extract conference key from file name
                                conferences.append(conference_key)
                                # Map the conference to its category using the dictionary
                                if conference_key in conference_categories:
                                    categories.append(conference_categories[conference_key])
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")
                        continue

        # Add the matched conferences and categories to the lists
        matched_conferences.append(", ".join(set(conferences)) if conferences else None)
        matched_categories.append(", ".join(set(categories)) if categories else None)

    # Add the matched conferences and categories to the faculty_data DataFrame
    faculty_data["Matched Conferences"] = matched_conferences
    faculty_data["Categories"] = matched_categories

    return faculty_data

def update_university_rankings(faculty_data, university_rankings_file, output_file):
    """
    Updates the university rankings CSV with edge weights for matched universities and fields.

    Args:
        faculty_data (pd.DataFrame): DataFrame containing faculty, matched conferences, and categories.
        university_rankings_file (str): Path to the university_rankings.csv file.
        output_file (str): Path to save the updated university rankings CSV.

    Returns:
        None
    """
    # Load the university rankings CSV
    university_rankings = pd.read_csv(university_rankings_file)

    # Iterate through each row in the faculty data
    for _, row in faculty_data.iterrows():
        university = row["Reference University"]
        categories = row["Categories"]
        edge_weight = row["Edge Weight"]

        if pd.notna(university) and pd.notna(categories):
            # Split categories into a list
            category_list = categories.split(", ")

            # Update the university rankings for each category
            for category in category_list:
                if category in university_rankings.columns:
                    university_rankings.loc[
                        university_rankings["University"] == university, category
                    ] += edge_weight

    # Save the updated university rankings to a new CSV file
    university_rankings.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Define file paths
    faculty_file = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/faculty/Scoring/faculty_full_names.csv'
    author_file = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/get_paper/author_universities_output.csv'
    paper_file = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/faculty/Scoring/author_paper_data.csv'
    conference_folder = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/get_paper/Conferences'
    university_rankings_file = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/public/university_rankings.csv'
    output_file = '/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/public/university_rankings.csv'

    # Check if the conference folder exists
    if not os.path.exists(conference_folder):
        print(f"Conference folder '{conference_folder}' does not exist.")
        exit(1)

    # Check if the university rankings file exists
    if not os.path.exists(university_rankings_file):
        print(f"University rankings file '{university_rankings_file}' does not exist.")
        exit(1)

    # Run the main function
    faculty_data = check_paper_in_conferences_with_categories(
        faculty_file, author_file, paper_file, conference_folder, CONFERENCES_CATEGORIES
    )
    
    # Update university rankings with edge weights
    update_university_rankings(faculty_data, university_rankings_file, output_file)
    print(f"Updated university rankings saved as {output_file}")


