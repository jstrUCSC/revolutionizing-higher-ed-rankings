import csv
import re

def clean_text(text):
    """Normalize text and remove problematic characters"""
    # Remove extra whitespace before punctuation
    text = re.sub(r'\s+([^\w\s])', r'\1', text)
    # Fix multiple spaces
    text = ' '.join(text.split())
    return text.strip()

def extract_authors(reference):
    """Extract authors from the reference string by splitting on commas and 'and'"""
    # Remove surrounding quotes if present
    reference = reference.strip('"')
    
    # If there are only two authors separated by "and" and no commas, handle this case
    if 'and' in reference and ',' not in reference:
        authors = [author.strip() for author in reference.split('and')]
        authors[-1] = authors[-1][:-1]  # Remove the period from the last author entry
        print(authors)
        return authors
    
    # Otherwise, split based on commas first
    authors = [author.strip() for author in reference.split(',')]
    
    # Handle the Oxford comma by checking the last segment for 'and'
    if 'and' in authors[-1]:
        # Remove 'and' from the last author entry
        authors[-1] = authors[-1].replace('and', '').strip()
    
    authors[-1] = authors[-1][:-1]  # Remove the period from the last author entry
    
    return authors

def extract_paper_title(paper):
    """Extract the paper title from the paper string."""
    # Find everything before the first period
    title_end_pos = paper.find('.')
    if title_end_pos != -1:
        title_section = paper[:title_end_pos].strip()
    else:
        title_section = paper.strip()

    # Remove surrounding quotes if present
    title_section = title_section.strip('"')
    return clean_text(title_section)

def read_publication_data(file_path):
    publication_data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'Reference' in row and 'Paper' in row:
                publication_data.append((row['Reference'], row['Paper']))
    return publication_data

def write_author_paper_data(publication_data, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Authors', 'Paper Title'])
        
        for reference, paper in publication_data:
            # Extract authors as a list
            authors = extract_authors(reference)
            paper_title = extract_paper_title(paper)
            
            # Write authors as a list and paper title
            if authors:
                writer.writerow([authors, paper_title])

def main():
    input_file_path = '../../llm/selected_references.csv'  # Adjust to your file path
    output_file_path = 'author_paper_data.csv'

    publication_data = read_publication_data(input_file_path)
    write_author_paper_data(publication_data, output_file_path)

if __name__ == '__main__':
    main()