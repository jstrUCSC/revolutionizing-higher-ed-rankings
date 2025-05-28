## Data Organization

### PDF Storage

- Research papers (PDFs) are stored in the `/publications` directory, organized by **conference name and year**.
- Example:
  ```
  /Publications/CVPR_2020/
  /Publications/NeurIPS_2020/
  ```

---

### CSV File Structure

| File Path | Description |
|-----------|-------------|
| `/faculty/Scoring/faculty_full_name.csv` | Contains each faculty member’s name and their **cumulative contribution score** derived from referenced papers. |
| `/public/university_rankings.csv` | Stores institution-level scores, **automatically updated** when `run_all.py` is executed. |
| `/data/author_university_output.csv` | Maps **individual authors to their affiliated universities**, used during the scoring phase. |
| `/public/country-info.csv` | Maps each university to its respective **country and continent**, used for regional breakdowns. |

> Only `university_rankings.csv` is automatically regenerated; others serve as static or semi-static metadata sources.

---

## Data Pipeline

1. **Paper Collection**  
   - PDFs are manually downloaded or retrieved using `get_paper/download_papers.py` (e.g., conference websites, Semantic Scholar).

2. **Metadata Extraction**  
   - Paper metadata (title, authors, conference, year) is extracted from the [CSRankings GitHub repository](https://github.com/emeryberger/CSrankings).
   - Additional parsing ensures compatibility with scoring logic (e.g., normalization of author names).

3. **LLM Analysis**  
   - Abstracts are passed to a **DeepSeek 8B Instruct** model via Hugging Face.
   - `get_references.py` extracts the **top 5 academic references** per paper.

4. **Scoring**  
   - Contributions are **weighted per reference** and distributed across identified authors and their institutions.
   - Implemented in `match_abbr_names.py` and `categorize_authors.py`.

5. **Data Storage**  
   - All processed outputs are written to structured CSVs in `/faculty/Scoring/` and `/public/`.
