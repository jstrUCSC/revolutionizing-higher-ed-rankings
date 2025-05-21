import pandas as pd
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
CURRENT_CONFERENCE = "ICML"
CONFERENCE_CATEGORIES = {
    'AAAI': 'Artificial Intelligence & Machine Learning', 'IJCAI': 'Artificial Intelligence & Machine Learning',
    'NeurIPS': 'Artificial Intelligence & Machine Learning', 'ICML': 'Artificial Intelligence & Machine Learning',
    'KDD': 'Data Science & Data Mining', 'ICDM': 'Data Science & Data Mining', 'SDM': 'Data Science & Data Mining',
    'CIKM': 'Data Science & Data Mining', 'WWW': 'Data Science & Data Mining',
    'CVPR': 'Computer Vision & Image Processing', 'ICCV': 'Computer Vision & Image Processing',
    'ECCV': 'Computer Vision & Image Processing',
    'ACL': 'Natural Language Processing', 'EMNLP': 'Natural Language Processing',
    'NAACL': 'Natural Language Processing', 'COLING': 'Natural Language Processing',
    'SOSP': 'Systems & Networking', 'OSDI': 'Systems & Networking', 'SIGCOMM': 'Systems & Networking',
    'NSDI': 'Systems & Networking', 'PODC': 'Systems & Networking', 'DISC': 'Systems & Networking',
    'FAST': 'Systems & Networking',
    'SIGMOD': 'Databases', 'VLDB': 'Databases', 'ICDE': 'Databases', 'EDBT': 'Databases',
    'CCS': 'Security & Privacy', 'USENIX Security': 'Security & Privacy', 'NDSS': 'Security & Privacy',
    'S&P (Oakland)': 'Security & Privacy',
    'CHI': 'Human Computer Interaction', 'UIST': 'Human Computer Interaction',
    'STOC': 'Theoretical Computer Science', 'FOCS': 'Theoretical Computer Science',
    'ICALP': 'Theoretical Computer Science', 'SODA': 'Theoretical Computer Science',
    'ICSE': 'Software Engineering', 'FSE': 'Software Engineering',
    'ASE': 'Software Engineering', 'ISSTA': 'Software Engineering',
    'SIGGRAPH': 'Computer Graphics & Virtual Reality', 'Eurographics': 'Computer Graphics & Virtual Reality',
    'IEEE VR': 'Computer Graphics & Virtual Reality',
    'QIP': 'Quantum Computing', 'TQC': 'Quantum Computing',
    'CoRL': 'Interdisciplinary Fields', 'AMIA': 'Interdisciplinary Fields', 'L@S': 'Interdisciplinary Fields',
}

CURRENT_CATEGORY = CONFERENCE_CATEGORIES.get(CURRENT_CONFERENCE, "Unknown")

# ----------------------------- FUNCTIONS -----------------------------

def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df

def assign_conference_and_university(faculty_file, author_file):
    """Assign conference/category and university by matching author names."""
    faculty_df = normalize_columns(pd.read_csv(faculty_file))
    author_df = normalize_columns(pd.read_csv(author_file))

    # Map authors to universities
    author_univ_map = author_df.drop_duplicates("Author").set_index("Author")["University"].to_dict()

    matched_rows = []
    for _, row in faculty_df.iterrows():
        name = row["Full Name"]
        university = author_univ_map.get(name)

        if university:
            print(f"[MATCH] Faculty '{name}' matched with university: {university}")
            row["University"] = university
            row["Matched Conferences"] = CURRENT_CONFERENCE
            row["Categories"] = CURRENT_CATEGORY
            matched_rows.append(row)
        else:
            print(f"[NO MATCH] Faculty '{name}' not found in author-university data (skipping)")

    return pd.DataFrame(matched_rows)

def update_university_rankings(faculty_df, rankings_file, output_file):
    """Increment university-category scores based on matched faculty only."""
    rankings_df = pd.read_csv(rankings_file)

    for _, row in faculty_df.iterrows():
        university = row["University"]
        categories = row["Categories"]
        score = row.get("Normalized Score", 1)  # Default to 1 if not present

        for category in categories.split(", "):
            if category in rankings_df.columns:
                rankings_df.loc[rankings_df["University"] == university, category] += score

    rankings_df.to_csv(output_file, index=False)

# ----------------------------- MAIN -----------------------------

if __name__ == "__main__":
    # Paths
    faculty_file = Path("../faculty/Scoring/faculty_full_names.csv")
    author_file = Path("../get_paper/author_universities_output.csv")
    rankings_file = Path("../public/university_rankings.csv")
    output_file = rankings_file  # Overwrite

    if not faculty_file.exists() or not author_file.exists():
        raise FileNotFoundError("Faculty or author file is missing.")
    if not rankings_file.exists():
        raise FileNotFoundError("University rankings file not found.")

    # Process
    faculty_df = assign_conference_and_university(faculty_file, author_file)
    update_university_rankings(faculty_df, rankings_file, output_file)

    print(f"\n✅ Faculty assigned to: {CURRENT_CONFERENCE} ({CURRENT_CATEGORY})")
    print(f"✅ University rankings updated and saved to {output_file}")