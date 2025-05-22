# Revolutionizing Higher Ed Rankings

**By:**  
- Jack Herbst – Frontend  
- Andrew Ketola – Backend  
- Baorong Luo – Frontend  
- Karl Mellinger – LLM / Graph Algorithm  
- Mainoah Muna – Backend  
- Lianghui Wang – LLM  

---

## Project Values

Current academic ranking systems, such as CS Rankings, focus heavily on publication count and reputation. This often incentivizes quantity over impactful work. Our project introduces a **new CS PhD university ranking system** that prioritizes **publication quality** using **LLMs** and **graph-based influence scoring**.  

**Key Values:**  
- Fair and transparent ranking  
- Focus on research impact, not just volume  
- Equitable treatment of collaborative work  
- Target users: prospective PhD students, academic advisors, funding agencies, and policymakers  

---

## Technical Documentation

### Architecture Overview

Our system is a **microservice-based pipeline** with the following core components:

1. **PDF Downloader**  
   - Fetches papers via URLs from a provided CSV dataset

2. **Author Extraction**  
   - Matches papers to authors using the **DBLP API**  
   - Resolves naming inconsistencies and affiliation data

3. **LLM Paper Scoring**  
   - Uses **deepseek-ai/DeepSeek-R1-Distill-Llama-8B** to evaluate paper quality by pulling the 5 most important references from the paper.  
   - Scoring pipeline parses LLM responses, calculates impact, and stores results

4. **Author Influence Graph**  
   - Nodes: Authors  
   - Edges: Citation links, weighted as `1 / number_of_authors`  
   - Author scores are aggregated to compute university scores  

### ⚙️ Backend & Data Flow

- Raw papers ➝ Author/University Mapping ➝ NLP Scoring ➝ Influence Graph ➝ Rankings
- Graph-based scoring ensures **collaborative fairness**
- Scoring is normalized across conferences

### 🗓️ Project Timeline

**Fall Term**
- Continent Filter (UI)  
- UI Redesign  
- LLM Research & Evaluation  
- LLM Integration Plan  

**Winter Term**
- Backend Scoring Infrastructure  
- DBLP Author Matching  
- PDF Downloader  
- Paper Scoring Testing  

**Spring Term**
- Final Feature Integration  
- Documentation Completion  

---

## End User Documentation

### Our Platform

**Explore the site:** *[Our Site](https://lianghui818.github.io/revolutionizing-higher-ed-rankings/public/)*  
Users can explore ranked university programs with filters and insights that go beyond traditional metrics.

---

### Features

- University Rankings by Quality of Research  
- Continent-Based Filters  
- Author Browsing  
- Transparent Scoring Breakdown  

---

### How to Use

**Visit the Site**  
Navigate to our hosted site to browse rankings.

**Apply Filters**  
Use the continent or research domain filter to narrow results.

**Inspect Rankings**  
Click on universities to view contributing authors and their scores.

**Understand the Score**  
Each score is backed by explainable metrics from paper analysis and graph-based aggregation.

---

### User Interface Help

The interface includes tooltips and intuitive design elements to guide users as they explore rankings, filters, and author data.

---

### Onboarding for New Users

First-time users can follow the "How to Use" section on the homepage or refer to this README for a complete walkthrough of features and functionality.

---

### Troubleshooting

Encountering issues?

- **Empty results?** Try resetting filters or expanding your criteria.
- **Slow load times?** Check your internet connection and refresh the page.
- **Other issues?** Visit the GitHub Issues tab to report bugs or request help.

---

### FAQ

**Q: How are university rankings calculated?**  
A: Rankings are based on research quality, not just quantity. We use a language model to assess paper and return the 5 most important references to showcase impact and construct a citation-based author influence graph.

**Q: Where does your data come from?**  
A: We use a curated dataset of CS papers from the top CS conferences and author metadata from DBLP, with PDF links for full-text analysis.

**Q: What makes this different from CS Rankings?**  
A: Unlike CS Rankings, we account for paper impact using LLM evaluation and fair contribution scoring via graph-based aggregation.

**Q: Why are some universities missing?**  
A: We only include institutions with verifiable publication data. Some universities may be excluded due to missing or ambiguous author affiliations.

**Q: How are co-authored papers scored?**  
A: Credit is distributed fairly among co-authors. In the influence graph, each citation link is weighted by 1 / number of authors.

**Q: Can I suggest improvements or report issues?**  
A: Yes! Please open an issue or pull request on our [GitHub repository](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings) to contribute or report bugs.

---

**All user and technical documentation is available in our GitHub repository README.**

---

## Project Resources

- [GitHub Repository](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings)
- [Documentation Page](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings/blob/main/README.md)

---

## Acknowledgments

- **Dr. Fuxin Li** – Associate Professor, Oregon State University  
- **Diji Yang** – PhD Student, University of California Santa Cruz  
