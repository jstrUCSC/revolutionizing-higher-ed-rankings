# Project Setup

## Prerequisites

- Python 3.10+

- Git

- Text editor (VS Code recommended)


## Installation Steps


1. **Clone the repository**

```bash
git clone https://github.com/Lianghui818/revolutionizing-higher-ed-rankings.git
cd your-repo
```
2. **(Optional) Setup Python virtual environment and install requirements:**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. **Run micro-service scripts manually to update CSV:**
```bash
#llm scripts
cd llm
python get_references.py
python at.py

#scoring/graphing scripts
cd faculty/Scoring
python match_abbr_names.py
cd faculty/Graph
python create_graph.py

#update csvs
cd utils
python categorize_authors.py
```
4. **Test frontend locally:**
```bash
cd public
python -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.