# Project Setup

## Prerequisites

- Python 3.10+

- Git

- Text editor 


## Installation Steps


1. **Clone the repository**

```bash
git clone https://github.com/Lianghui818/revolutionizing-higher-ed-rankings.git
cd revolutionizing-higher-ed-rankings
```
2. **(Optional) Setup Python virtual environment and install requirements:**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. **Run python script to run llm and update csv's:**
```bash
cd utils
python run_all.py <Conference Directory>
# ie. python run_all.py CVPR_2020
```
Might take 2-3 days to run depending on amount of pdfs in directory

3. **Test frontend locally:**
```bash
cd public
python -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.
