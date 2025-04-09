import pandas as pd

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

import pandas as pd

def filter_faculty_with_universities(faculty_file, author_file):
    """
    Filters faculty data to include only those with university ties.

    Args:
        faculty_file (str): Path to the faculty_full_names.csv file.
        author_file (str): Path to the author_universities_output.csv file.

    Returns:
        pd.DataFrame: Filtered DataFrame with university ties.
    """
    # Load the faculty_full_names.csv file
    faculty_data = pd.read_csv(faculty_file)

    # Load the author_universities_output.csv file
    author_data = pd.read_csv(author_file)

    # Create a dictionary to map authors to their universities
    author_to_university = author_data.drop_duplicates(subset=["Author"]).set_index("Author")["University"].to_dict()

    # Add university information for the "Reference Name" column in the faculty_data DataFrame
    faculty_data["Reference University"] = faculty_data["Reference Name"].map(author_to_university)

    # Drop references without university ties
    faculty_data = faculty_data.dropna(subset=["Reference University"])

    return faculty_data


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

