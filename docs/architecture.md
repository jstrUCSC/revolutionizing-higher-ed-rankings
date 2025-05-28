# Project Documentation

## Overview

- Static frontend served locally using Python’s HTTP server.
- Backend consists of manually run Python scripts that process papers, interact with LLM, and update CSV data files.
- All data storage uses CSV files in respective data folders.
- Frontend fetches CSV data dynamically and renders rankings and graphs client-side.

---

### Workflow

1. Run backend scripts to process publications and generate ranking data.
2. Frontend loads CSV files and visualizes rankings and relationships.
3. No persistent server or database; all flat files.

---

## LLM

- We use the **DeepSeek LLM (8B Instruct)** from Hugging Face to extract the **top 5 academic references** from each publication abstract.
- This logic is implemented in `llm/get_references.py` using the `transformers` library.
- The retrieved references are used to assess academic influence and contribute to university scoring.

---

## Backend

- Python scripts in `/llm`, `/faculty`, `/utils`.
- Key scripts:
  - `at.py`: converts llm response to json for parsing
  - `get_references.py`: Calls LLM, extracts references.
  - `match_abbr_names.py`: Computes scores for each author that was referenced.
  - `categorize_authors.py`: Utility functions for CSV operations to match authors to universities then update rankings.
- Run programs all together utilizing backend script in `utils\run_all.py`.

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
