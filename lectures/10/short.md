# Lecture 10: Experimentation, Research Design, and Distributed Computing in Health Data Science

## Course Recap & Integration
- Summary of previous lectures: Python fundamentals, EDA, ML, visualization, reporting

## 0. Introduction & Review
- Purpose: connect distributed computing and experimental methodology  
- Transition from visualization/reporting to scalable workflows and research design

## 1. Distributed Computing & Scaling in Health Data Science

**1.1. Motivation & Health Use Cases**
- Genomics cohorts, imaging archives, EHR warehouses, wearable streams  
- Gene expression in transplant patients: high-dimensional matrix → scaling needs

**1.2. Compute Patterns**
- Threads vs. Processes (I/O-bound vs. CPU-bound tasks)  
- Parallelism vs. Concurrency

**1.3. Architectures & Scaling**
- Single-machine: multithreading, multiprocessing (GIL considerations)  
- Cluster: SGE/SLURM on UCSF Wynton  
- Cloud: Kubernetes, Spark, serverless functions

**1.4. Python Libraries & Tools**
- multiprocessing & concurrent.futures  
- Dask, Ray, PySpark

**1.5. Workflow Orchestration & Containers**
- Airflow, Prefect, Nextflow for pipeline management  
- Docker & Singularity for reproducible HPC jobs

## 🛠️ Demo Break 1: Parallel Computing in Practice
**Objective:** Run a parallel patient-analysis locally and submit to SLURM  
**Steps:** multiprocessing.Pool demo; adapt code to a SLURM script  
**Success Criteria:** Observe local speedup and create a basic cluster job script

## 2. Experimental Design & Analysis in Health Data Science

**2.1. Foundations of Experimental Design**
- Causation vs. Correlation & Bradford Hill criteria  
- Confounding variables & control methods  
- Randomized Controlled Trials vs. observational cohort/case-control designs

**2.2. Statistical Methods & Python Tools**
- t-tests, ANOVA, non-parametric tests (scipy.stats)  
- Generalized linear models (statsmodels)  
- Multiple testing correction & power analysis

**2.3. Ethics & Compliance**
- IRB processes, informed consent essentials  
- HIPAA, GDPR, secure data handling

**2.4. Twofold Research Approach**
- Engage subject-matter experts (PIs, literature review)  
- Exploratory Data Analysis on gene expression transplant dataset  
- Brainstorm study ideas from SME input and EDA insights

## 🛠️ Demo Break 2: EDA & Project Generation Walkthrough
**Objective:** Explore gene expression data and propose research questions  
**Steps:** Load dataset; summary statistics; key visualizations; sketch study designs  
**Success Criteria:** Define two project outlines leveraging experimental design and scaling

## 3. End-to-End Health Data Science Project Application

**3.1. CRISP-DM for Health Data Science**
- Problem definition & stakeholder analysis  
- Data acquisition & quality assessment  
- Exploratory Data Analysis & hypothesis generation  
- Modeling & intervention design  
- Evaluation & iteration  
- Communication, documentation, reproducibility, deployment

**3.2. Illustrative Example: Gene Expression in Transplant Patients**
- Walk through CRISP-DM phases using the example dataset  
- Highlight where compute scaling and experimental design intersect

**3.3. Additional Project Examples (Brief)**
- Clinical trial A/B pipeline  
- EHR cohort study workflow  
- Imaging ML pipeline