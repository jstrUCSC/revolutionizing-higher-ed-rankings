# Automation of Extended Paper Processing and Graph Creation

## Tasks

- Extend the existing automated paper downloading system to include additional conferences beyond the current set.

## Goals

- Seamlessly integrate new conferences into the automated download system.
- Fully automate end-to-end processing from download to graph generation.
- Implement scheduling or triggers to run the complete pipeline regularly or on-demand.
- Ensure proper logging and error handling across all steps.

## Implementation Ideas

- Update configurations or parameters to include new conferences in the downloader.
- Use a master orchestration script or workflow tool to chain paper downloading, LLM scoring, and graph creation.
- Set up cron jobs or event-driven triggers to automate pipeline execution.
