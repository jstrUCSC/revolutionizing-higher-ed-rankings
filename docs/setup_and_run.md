# Project Setup

## Prerequisites

Make sure you have the following installed on your machine:

- **Python 3.10+**
- **Git**
- **A text editor or IDE** (e.g., VS Code, PyCharm, etc.)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Lianghui818/revolutionizing-higher-ed-rankings.git
cd revolutionizing-higher-ed-rankings
```

### 2. (Optional but recommended) Setup python virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
### 3. Install python dependencies
```bash
pip install -r requirements.txt
```
### 4. Download papers 
```bash
cd get_papers
python download_papers.py <Conference Paper CSV>

# Example:
python download_papers.py CVPR_2020_papers.csv
```

Papers that are downloaded will be strored in the `Publications/` divided by conference and year.

### 5. Run backend script to run papers through LLM and update CSVs
```bash
cd utils
python run_all.py <Conference Paper directory>

# Example:
python run_all.py CVPR_2020
```

**Note:** Make sure to change this line of code in `utils/categorize_authors.py` depending on which conference papers are being ran. 
```python
CURRENT_CONFERENCE = "CVPR"
```

### 6. Test frontend locally
Serve the public folder using Python’s built-in HTTP server:
```bash
cd ../public
python -m http.server 8000
```
Then open your browser and navigate to:
http://localhost:8000
