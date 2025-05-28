## Deployment

### Frontend

- The frontend is composed of static files (`index.html`, CSS, JS), hosted via **GitHub Pages** from the `main` branch.
- No build step is needed — any changes pushed to the GitHub Pages branch are immediately reflected live.
- To preview locally:
  ```bash
  cd public
  python -m http.server 8000
  ```

### Backend

- Backend scripts are **run manually**
- To execute all backend processes:
  ```bash
  python utils/run_all.py
  ```
- Designed for execution on a **local machine or a small-scale server**

### Data Storage

- **Output data (e.g., university rankings)** is stored as flat CSV files in:
  ```
  /public/university_rankings.csv
  ```
- **Metadata files** (e.g., parsed abstracts, author matches) are stored in:
  ```
  /get_papers/
  /data/
  /faculty/Scoring/
  ```
- All CSVs are Git-tracked to preserve version history and ensure reproducibility.

### Infrastructure

- **No containerization (e.g., Docker)** or orchestration (e.g., Kubernetes) is used or required.
- **No cloud database** or remote storage dependency — ideal for lightweight, file-based workflows.
- Compatible with basic web hosting, on-prem servers, and GitHub-hosted sites.
