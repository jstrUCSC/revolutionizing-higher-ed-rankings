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
   - Uses **deepseek-ai/DeepSeek-R1-Distill-Llama-8B** to evaluate paper quality  
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
- Country Filter (UI)  
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

> **Explore the site:** [Our Site](https://lianghui818.github.io/revolutionizing-higher-ed-rankings/public)

Users can explore ranked university programs with filters and insights that go beyond traditional metrics.

### Features

- **University Rankings by Quality of Research**  
- **Country-Based Filters**  
- **Author Browsing**  
- **Transparent Scoring Breakdown**

### How to Use

1. **Visit the Site**  
   Navigate to our hosted site to browse rankings.

2. **Apply Filters**  
   Use the country filter or research domain filter to narrow results.

3. **Inspect Rankings**  
   Click on universities to view contributing authors and their scores.

4. **Understand the Score**  
   Each score is backed by explainable metrics from paper analysis and graph-based aggregation.

---

## 📂 Project Resources

- [GitHub Repository](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings)
- [Documentation Page](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings/blob/main/README.md)

---

## Acknowledgments

- **Dr. Fuxin Li** – Associate Professor, Oregon State University  
  *fuxin.li@oregonstate.edu*  
- **Diji Yang** – PhD Student, University of California Santa Cruz  
