## Data

- PDFs are downloaded and stored in `/publications` folder under their specific conference and year.
  - ie. CVPR 2020 papers are stored in `/publications/CVPR_2020`.

### CSV Structure

- `/faculty/Scoring/faculty_full_name.csv` — Contains authors and their cumulative scores from processed publications.
- `/public/university_rankings.csv` — Contains scores assigned to each institution.
- `/data/author_university_output.csv` — Stores authors and their affiliate universities.
- `/public/country-info.csv`  — Stores the university and their respective continent. 

university_rankings.csv is updated automatically when the backend scrip is run. All the other .csv files are utilized to store metadata. 

### Data Pipeline

1. **Paper Collection:** PDFs are gathered manually or through scraping tools.
2. **Metadata Extraction:** Data is sourced from the CSRANKINGS github repository.
3. **LLM Analysis:** Top 5 references are pulled from each paper using an LLM model.
4. **Scoring:** Contributions are weighted and distributed across authors and institutions.
5. **Storage:** All results are written back to the corresponding CSV files.

### Notes

- Files are version-controlled with Git to track changes.
- CSVs support easy review/editing with spreadsheet tools.
