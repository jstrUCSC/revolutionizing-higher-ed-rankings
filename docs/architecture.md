### Overview

- Static frontend served locally using Python’s HTTP server.
- Backend consists of manually run Python scripts that process papers, interact with LLM, and update CSV data files.
- All data storage uses CSV files in respective data folders
- Frontend fetches CSV data dynamically and renders rankings and graphs client-side.

### Workflow

1. Run backend scripts to process publications and generate ranking data.
2. Frontend loads CSV files and visualizes rankings and relationships.
3. No persistent server or database; all flat files.
---

## Backend

- Python scripts in `/llm`,  `/faculty`,  `/utils`.
- Key scripts:
  - `get_references.py`: Calls LLM, extracts references.
  - `match_abbr_names.py`: Computes scores for each author that was referenced.
  - `categorize_authors.py`: Utility functions for CSV operations to match authors to universities then update rankings.

- Run scripts manually; no live backend server.

---

## Frontend

- Pure HTML, CSS, JavaScript; no frameworks.
- Tested locally using Python's simple HTTP server.
- Fetches CSV files and renders institution rankings.
- Features dynamic filtering and graph visualization planned.

To run:
```bash
cd frontend
python -m http.server 8000
```
Open [http://localhost:8000](http://localhost:8000).

---