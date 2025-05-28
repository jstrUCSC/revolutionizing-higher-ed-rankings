# Project Documentation

## Overview

- **Frontend:** A static frontend built with pure HTML, CSS, and JavaScript, served locally using Python’s built-in HTTP server.
- **Backend:** A suite of manually executed Python scripts for processing academic papers, interacting with an LLM, and updating structured CSV data files.
- **Data Storage:** All data is stored in flat CSV files organized within data/ subdirectories.
- **Client-Side Logic:** The frontend dynamically fetches CSV data to render rankings, visualizations, and institution relationships without a backend server or database.
---

## Project Structure

```
.
├── llm/
│   ├── get_references.py
│   ├── at.py
├── faculty
|   └── /Scoring
│         └── match_abbr_names.py
|   └── /Graph
|        └── create_graph.py
|        └── author_scores.csv
|        └── author_total.py
├── utils/
|   └── data/
|        └── (CSRankings data used to populate CSVs)
│   └── run_all.py
│   └── categorize_authors.py
├── public/
│   └── (HTML/CSS/JS files)
└── README.md
└── requirement.txt
```
---

### Workflow

1.	Run backend scripts to process raw publication data and generate updated scores and rankings.
2.	Output CSV files are placed in the appropriate data/ folders.
3.	Frontend loads the generated CSV files and visualizes university rankings and reference networks.
4.	The system is entirely file-based with no persistent server or database backend.
---

## LLM

We use the **DeepSeek LLM (8B Instruct)** hosted on Hugging Face to extract the top 5 academic references from each publication abstract.
- Implemented in llm/get_references.py using the transformers library from Hugging Face.
- Extracted references contribute to the academic influence scoring model used to rank universities.

---

## Backend

Directory Structure: Core logic is divided among /llm, /faculty, and /utils folders.
- Key Scripts:
- `llm/get_references.py`: Sends publication abstracts to the LLM and extracts reference data.
- `llm/at.py`: Parses and formats LLM responses into JSON for further processing.
- `faculty/Scoring/match_abbr_names.py`: Matches references to abbreviated author names and computes   institutional scores.
-  `utils/categorize_authors.py`: Maps authors to universities and updates CSV-based ranking tables.
- `utils/run_all.py`: Master script that executes all major backend steps in order.
    
**Execution:** Run utils/run_all.py to trigger the full processing pipeline.

---

## Frontend
- Built using vanilla HTML, CSS, and JavaScript.
- No external frameworks or dependencies.
- Features:
  - Fetches CSV files on page load.
  - Dynamically renders institutional rankings and scoring metrics.
  - Planned support for interactive filtering and graph-based visualization of academic relationships.

To run locally:
```bash
cd public
python -m http.server 8000
```
