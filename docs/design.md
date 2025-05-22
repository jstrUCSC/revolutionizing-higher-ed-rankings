# Project Team

- **Mainoah Muna** - [munam@oregonstate.edu](mailto:munam@oregonstate.edu)  
- **Lianghui Wang** - [wangl9@oregonstate.edu](mailto:wangl9@oregonstate.edu)  
- **Karl Mellinger** - [mellinka@oregonstate.edu](mailto:mellinka@oregonstate.edu)  
- **Baorong Luo** - [luoba@oregonstate.edu](mailto:luoba@oregonstate.edu)  
- **Andrew Ketola** - [ketolaa@oregonstate.edu](mailto:ketolaa@oregonstate.edu)  
- **John Herbst** - [herbstj@oregonstate.edu](mailto:herbstj@oregonstate.edu)  

# Revolutionizing Higher Ed Rankings

**Course:** CS462  
**Document:** Feedback Review

## Feedback Review

Our project partner recommends retaining our existing CSV-based storage system instead of transitioning to an SQL database. Our current setup, which relies on CSV files, has proven effective for managing data due to its simplicity and direct compatibility with widely used tools like Excel and text editors. This approach minimizes the learning curve for stakeholders already familiar with CSV workflows and avoids the costs and complexity of adopting SQL, such as licensing fees, server setup, or database administration.

Since our system already operates smoothly with CSV files, migrating to SQL would introduce unnecessary overhead, including schema design, data migration, and retraining efforts. CSV’s flexibility allows rapid edits and version control integration, which aligns with our iterative processes and small-to-medium dataset requirements. While SQL offers advantages for complex queries or large-scale data, these features are not critical to our current operations, and the existing CSV framework adequately meets our needs without added maintenance or disruption.

We recommend continuing with CSV to preserve efficiency and cost-effectiveness, revisiting SQL only if future scalability or advanced functionality demands arise.

---

## Changelog

| Date       | Version | Change Description                                                                 | Reason                                                                                                      | Author/Initiator   |
|------------|---------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------|
| 2024-11-03 | 1.0     | Initial version submitted fall term.                                                | Creation                                                                                                    | Team               |
| 2025-02-10 | 1.0.1   | Duplicated for winter term updates, removed activities from last submission.        | Copy of document created for new term.                                                                      | Mainoah Muna       |
| 2025-03-01 | 1.0.2   | Data extraction and processing will be done with the current CSV style and not SQLite. | No need for an extra database, should have enough local storage for these files.                           | Karl Mellinger     |
| 2025-03-01 | 1.0.3   | Removed section about containerizing with Docker                                     | No need for containerization; project doesn’t deploy on cloud platforms or require user-specific dependencies. | Karl Mellinger     |
| 2025-03-01 | 1.0.4   | - Removed cloud platform querying explanation  <br> - Added Python to “Technologies Used” <br> - Minor rewording to “Data Extraction and Processing” <br> - Changed course number from 461 to 462 | Python used for publication retrieval and LLM tasks. Cloud querying efficiency unaffected by CSV usage.    | Andrew Ketola      |
| 2025-03-02 | 1.0.5   | - Updated visual of system design to have correct CSV storage component              | We are not utilizing SQL. Team decided to stick with CSV.                                                  | Mainoah Muna       |

---

# Higher Ed Ranking Design Document

## Introduction

We are creating a ranking system for computer science research publications that focuses on fair and equitable rankings, rather than exclusively focusing on the quantity of publications from an institution. This document is an overview of the design decisions made to facilitate this goal, and the steps taken to integrate this system with an existing codebase. Our goal for this project is to create an equitable ranking system with a clean user interface that can support future revisions to its algorithm and LLM integration.

---

## Design Overview and Goals

The CS PhD ranking system is built upon several interconnected components designed to achieve the project’s overarching goals: creating a content-focused, equitable ranking system for computer science institutions. Below are the core components and their roles in meeting these objectives.

### Core Components

#### Large Language Model Integration

- **Role:** Evaluates the quality and impact of research papers by analyzing their content.  
- **Functionality:** Processes paper content to extract insights about novelty, technical depth, and field relevance.  
- **Alignment with Goals:** Provides an objective, content-driven assessment that emphasizes research quality over quantity.

#### Data Extraction and Processing

- **Role:** Automates the retrieval and structuring of research paper data from publication sources.  
- **Functionality:** Extracts papers and their metadata such as titles, authors, affiliations, and publication venues.  
- **Alignment with Goals:** Ensures accurate and comprehensive data collection for fair and unbiased rankings.

#### Scoring Algorithm

- **Role:** Converts insights from the LLM and metadata into rankings for institutions.  
- **Functionality:** The LLM analyzes each research paper to extract its top 5 most influential references. These references are used to infer the academic impact of the paper, forming the basis for scoring. A graph is then constructed where nodes represent institutions and edges capture shared contributions through co-authorship and citation influence. Each institution receives a portion of the paper’s impact score based on the affiliation of the authors and the prominence of the cited references. This scoring is proportional, ensuring credit is distributed fairly across all contributing institutions based on both authorship and cited influence.
- **Alignment with Goals:** This approach enhances fairness by focusing on both the quality (via LLM-derived references) and collaboration (via graph modeling). It prevents inflation through quantity alone and provides a more accurate and equitable reflection of research significance and institutional contribution.

#### Backend Infrastructure

- **Role:** Powers the system’s core functionalities.  
- **Functionality:** Manages interactions between frontend, LLM, and CSV storage.  
- **Alignment with Goals:** Supports seamless performance, scalability, and data reliability.

#### Frontend User Interface (UI/UX)

- **Role:** Provides access to explore rankings.  
- **Functionality:** Displays rankings with filters and transparency into calculation metrics.  
- **Alignment with Goals:** Improves user accessibility and global usability.

#### Affiliation and Contribution Parsing

- **Role:** Associates authors with institutions.  
- **Functionality:** Parses and distributes institutional credit based on contribution.  
- **Alignment with Goals:** Ensures collaborative contributions are accurately recognized.

---

## Key Design Decisions

### Modular Architecture

- **Description:** System is divided into modules: Frontend, Backend, LLM, CSV Storage, Data Sources.  
- **Rationale:** Promotes maintainability and independent development.

### Backend-Centric Workflow

- **Description:** Backend acts as central hub.  
- **Rationale:** Enables robust data flow and efficient processing.

### LLM Integration

- **Description:** Analyzes pdfs and pull top 5 references for scoring.  
- **Rationale:** Allows qualitative evaluations of research.

### CSV-Based Metadata Storage

- **Description:** Stores metadata and scores in CSV format.  
- **Rationale:** Lightweight and maintainable for current scale.

### Transparent Frontend

- **Description:** UI with filters and ranking explanations.  
- **Rationale:** Enhances usability and builds trust.

### Data Flow Automation

- **Description:** Automated pipelines for data handling.  
- **Rationale:** Reduces manual work and keeps system current.

### Scalable Deployment

- **Description:** Compatible with local and cloud infrastructure.  
- **Rationale:** Supports future expansion and distributed processing.

---

## Existing System Integration

This project builds on a previous capstone team’s work. Key integration efforts include:

### Integration of LLM

- **Output Format:** Standardized for backend processing.  
- **APIs:** Handle real-time communication between LLM and backend.

### Integration of Ranking System

- **Backend Logic:** Updated for impact-based scoring.  
- **UI Enhancements:** Filters and sorting by continent, country, institution.

### UI Enhancements

- **Researcher Profiles:** Show impact metrics.  
- **Global Views:** Display country/continent-level rankings.

---

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript (modifying existing structure).  
- **Backend:** Python.  
- **LLM Integration:** Deepseek R1 via LLM modular microservice.  
- **Storage:** CSV-based local storage.

---

## Design Considerations

### Performance

- **Approach:** CSV storage for efficient queries. 

### Scalability

- **Flexibility:** CSV supports growth with minimal overhead.  
- **Data Volume:** Scalable for additional papers and conferences.

### Security

- **HTTPS Communication:** Ensures data confidentiality.  
- **Input Sanitization:** Prevents injection attacks.  
- **Access Control:** Backend-exclusive CSV access.

---