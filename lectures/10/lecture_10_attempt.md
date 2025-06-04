# Lecture 10: Experimentation, Research Design, and Distributed Computing in Health Data Science

## 0. Introduction & Review

We've built a comprehensive toolkit over the past nine lectures, progressing from Python fundamentals through advanced machine learning applications. Today we integrate these skills with experimental methodology and scalable computing - two critical elements for conducting robust health data science research.

**Course Journey Recap:**
- **Development environment** and Python fundamentals gave us the foundation
- **Polars and SQL** taught us to handle large datasets efficiently  
- **Machine learning and deep learning** provided predictive modeling capabilities
- **Computer vision and NLP** expanded our ability to work with complex data types
- **Visualization and reporting** enabled us to communicate findings effectively

Today's goal is to connect these technical skills with proper study design and the computational infrastructure needed for real-world health research. We'll use gene expression analysis in transplant patients as our unifying example throughout.

## 1. Distributed Computing & Scaling in Health Data Science

### 1.1. Motivation & Health Use Cases

Modern health datasets routinely exceed the capacity of a single machine. When your laptop takes hours to process a dataset, or when you run out of memory entirely, it's time to think about distributed computing.

**Real-world scale examples** that demand distributed approaches:
- **Genomics cohorts:** UK Biobank (500,000+ participants), All of Us (1 million+ participants)
- **Imaging archives:** RadImageNet (2+ million medical images), hospital PACS systems
- **EHR warehouses:** Epic Cosmos (180+ million patient records), multi-hospital networks
- **Wearable streams:** Apple Health Study (400,000+ participants), continuous monitoring data

**Gene Expression in Transplant Patients Scenario:** Imagine analyzing RNA-seq data from 10,000 transplant recipients across multiple hospitals. Each sample generates ~50,000 gene expression measurements, creating a 10,000 × 50,000 matrix requiring several gigabytes of RAM just to load. Add in clinical covariates, longitudinal follow-up data, and the need for cross-validation across hospital sites, and you quickly exceed single-machine capabilities.

### 1.2. Compute Patterns

Understanding different concurrency models helps you choose the right approach for your specific health computing task.

**Threads vs. Processes (I/O-bound vs. CPU-bound tasks):**

**Threads (Shared Memory)** share memory space and excel at I/O-bound tasks where you're waiting for data transfers rather than crunching numbers. Think downloading DICOM images from multiple hospital PACS systems simultaneously - while one thread waits for a network response, others can continue working.

**Processes (Isolated Memory)** have isolated memory spaces and bypass Python's Global Interpreter Lock (GIL), making them ideal for CPU-intensive work like image processing or statistical computations.

**Parallelism vs. Concurrency:**

**Parallelism (True Simultaneous)** means multiple CPU cores genuinely work simultaneously on different parts of your problem. Picture 8 cores each analyzing a different patient's genomic data at the exact same time.

**Concurrency (Interleaved Tasks)** means rapidly switching between tasks on fewer cores, creating the illusion of simultaneity. One core might download patient data while "simultaneously" preprocessing the previous patient's images.

### 1.3. Architectures & Scaling

Health computing architectures have evolved from single workstations to sophisticated distributed systems.

**Single-machine: multithreading, multiprocessing (GIL considerations):**
- **Multi-threading:** Concurrent DICOM downloads, database queries (limited by GIL for CPU work)
- **Multi-processing:** Parallel image processing, model training (bypasses GIL)
- **Memory mapping:** Working with datasets larger than RAM

**Cluster: SGE/SLURM on UCSF Wynton:**
- **UCSF Wynton:** 4,000+ cores for health research, GPU clusters
- **Job scheduling:** Submit parallel tasks, resource allocation
- **Use cases:** Embarrassingly parallel health data analysis, genomics pipelines

**Cloud: Kubernetes, Spark, serverless functions:**
- **Auto-scaling:** Scale up during flu season for symptom tracking
- **Containers:** Reproducible analysis environments across hospitals  
- **Serverless:** Trigger analysis when new patient data arrives

### 1.4. Python Libraries & Tools

**multiprocessing & concurrent.futures:**

**Reference Card: `multiprocessing.Pool`**
- **Function:** `multiprocessing.Pool(processes=None)`
- **Purpose:** Create a pool of worker processes for CPU-bound parallel execution
- **Key Parameters:**
    - `processes`: (Optional, default=None) Number of worker processes. If None, uses `os.cpu_count()`
    - `initializer`: (Optional, default=None) Function to run when each worker process starts
    - `maxtasksperchild`: (Optional, default=None) Number of tasks a worker completes before restarting

**Key Methods:**
- `map(func, iterable)`: Apply function to each item in parallel, preserving order
- `starmap(func, iterable)`: Like map but unpacks argument tuples

```python
import multiprocessing as mp
import time

def analyze_patient(patient_id):
    # Simulate patient analysis
    time.sleep(0.1)
    return f"Patient {patient_id} analyzed"

if __name__ == "__main__":
    patient_ids = list(range(100))
    
    # Sequential processing
    start_time = time.time()
    results_seq = [analyze_patient(pid) for pid in patient_ids]
    seq_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        results_par = pool.map(analyze_patient, patient_ids)
    par_time = time.time() - start_time
    
    print(f"Sequential: {seq_time:.2f}s, Parallel: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
```

**Dask, Ray, PySpark:**

**Reference Card: `dask.delayed`**
- **Function:** `dask.delayed(func)`
- **Purpose:** Build lazy task graphs for complex workflows, then execute in parallel
- **Key Features:**
    - **Lazy evaluation:** Tasks aren't executed until you call `compute()`
    - **Familiar APIs:** Works with NumPy, Pandas operations
    - **Memory management:** Handles larger-than-memory datasets through chunking

```python
import dask
from dask import delayed

@delayed
def load_patient_data(patient_id):
    # Load patient data from file/database
    return f"data_for_patient_{patient_id}"

@delayed
def preprocess_data(data):
    # Clean and preprocess
    return f"processed_{data}"

@delayed
def analyze_data(data):
    # Perform analysis
    return f"analysis_{data}"

# Build computation graph
patient_ids = [1, 2, 3, 4, 5]
results = []
for pid in patient_ids:
    data = load_patient_data(pid)
    processed = preprocess_data(data)
    analysis = analyze_data(processed)
    results.append(analysis)

# Execute all tasks in parallel
final_results = dask.compute(*results)
print(final_results)
```

### 1.5. Workflow Orchestration & Containers

**Airflow, Prefect, Nextflow for pipeline management:**

Modern health data pipelines require orchestration tools to manage complex workflows with dependencies, error handling, and monitoring.

**Apache Airflow** provides directed acyclic graphs (DAGs) for workflow management:
- **Health use case:** Daily EHR data processing, weekly model retraining
- **Features:** Web UI, scheduling, failure notifications
- **Example:** Patient cohort identification → feature extraction → model training → validation

**Prefect** offers a Python-native approach to workflow orchestration:
- **Health use case:** Clinical trial data pipeline, genomics analysis workflows
- **Features:** Dynamic workflows, parameterization, cloud execution

**Nextflow** specializes in bioinformatics and computational pipelines:
- **Health use case:** Genomics variant calling, RNA-seq analysis
- **Features:** Portable across compute environments, automatic parallelization

**Docker & Singularity for reproducible HPC jobs:**

Containers ensure consistent environments across different computing systems, critical for reproducible health research.

**Docker** for development and cloud deployment:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "analyze_patients.py"]
```

**Singularity** for HPC environments like Wynton:
- **Security:** Runs without root privileges on shared clusters
- **Performance:** Optimized for HPC workloads
- **Integration:** Works with SLURM job schedulers

```bash
# Build container
singularity build analysis.sif analysis.def

# Submit to SLURM
#SBATCH --container analysis.sif
#SBATCH --cpus-per-task=8
python analyze_large_cohort.py
```

## 🛠️ Demo Break 1: Parallel Computing in Practice

**Objective:** Run a parallel patient-analysis locally and submit to SLURM

**Steps:**
- Use `multiprocessing.Pool` to parallelize a simple analysis across multiple patient files
- Measure speedup compared to sequential processing  
- Write a SLURM job script that could run the same analysis on Wynton
- Identify resource requirements (cores, memory, time) for the job

**Success Criteria:** 
- Observe measurable speedup from parallel processing
- Generate a runnable SLURM script with appropriate resource requests
- Understand when parallelization helps vs. hurts performance

---

## 2. Experimental Design & Analysis in Health Data Science

Rigorous experimental design separates robust health research from wishful thinking. The most sophisticated machine learning model can't overcome fundamental flaws in study design or data collection.

### 2.1. Foundations of Experimental Design

**Causation vs. Correlation & Bradford Hill criteria:**

**Correlation vs. Causation** remains the central challenge in health research. Just because coffee drinkers have lower rates of heart disease doesn't mean coffee prevents heart attacks - lifestyle factors like exercise and diet likely confound this relationship.

The **Bradford Hill criteria** provide a framework for assessing causal relationships:
- **Strength of association:** Larger effect sizes suggest causation
- **Dose-response relationship:** More exposure leads to greater effect  
- **Temporal sequence:** Cause must precede effect
- **Biological plausibility:** Mechanism makes scientific sense

**Confounding variables & control methods:**

**Confounding variables** affect both treatment assignment and outcomes, creating spurious associations. Control strategies help isolate causal effects:
- **Randomization** balances known and unknown confounders across groups
- **Matching** pairs similar patients receiving different treatments
- **Stratification** analyzes subgroups separately to control for key variables
- **Regression adjustment** statistically controls for measured confounders

**Randomized Controlled Trials vs. observational cohort/case-control designs:**

**Randomized Controlled Trials (RCTs):**
- **Design Principles:** Randomization, control groups, blinding
- **Randomization Methods:** Simple, block, stratified, adaptive
- **Gold standard** for causal inference but expensive and time-consuming

**Observational Studies:**
- **Cohort Studies:** Follow participants forward (prospective) or backward (retrospective) in time
- **Case-Control Studies:** Compare cases (with disease) to controls (without disease)
- **Cross-Sectional Studies:** Snapshot of population at single time point
- **Faster and cheaper** but require careful attention to confounding

### 2.2. Statistical Methods & Python Tools

**t-tests, ANOVA, non-parametric tests (scipy.stats):**

Choosing appropriate statistical tests depends on your data type, sample size, and research question.

**Reference Card: `scipy.stats.ttest_ind`**
- **Function:** `scipy.stats.ttest_ind(a, b, equal_var=True)`
- **Purpose:** Compare means between two independent groups
- **Key Parameters:**
    - `a, b`: (Required) Sample data for the two groups being compared
    - `equal_var`: (Optional, default=True) Assume equal population variances. Set to False for Welch's t-test
    - `alternative`: (Optional, default='two-sided') Alternative hypothesis ('two-sided', 'less', 'greater')

```python
import scipy.stats as stats
import numpy as np

# Health example: Compare gene expression between rejection/no-rejection groups
rejection_group = [10.5, 15.2, 12.8, 8.9, 14.1, 11.3, 9.7, 13.2]
no_rejection_group = [6.2, 5.8, 7.3, 4.1, 6.9, 5.4, 4.8, 6.1]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(rejection_group, no_rejection_group)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Significant difference: {p_value < 0.05}")

# Check assumptions
print(f"Rejection group normality: {stats.shapiro(rejection_group).pvalue:.3f}")
print(f"No-rejection group normality: {stats.shapiro(no_rejection_group).pvalue:.3f}")
```

**Generalized linear models (statsmodels):**

**GLM components** extend linear regression to various outcome types:
- **Linear predictor:** β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
- **Link function:** Connects linear predictor to expected outcome
- **Family distribution:** Specifies outcome distribution (normal, binomial, Poisson)

**Reference Card: `statsmodels.formula.api`**

```python
import statsmodels.formula.api as smf
import pandas as pd

# Health example: Logistic regression for transplant rejection prediction
data = pd.DataFrame({
    'rejection': [0, 1, 0, 1, 1, 0, 1, 0],
    'age': [45, 62, 38, 58, 67, 41, 55, 49],
    'hla_mismatch': [2, 4, 1, 3, 5, 2, 4, 1],
    'gene_expression': [6.2, 12.8, 5.9, 11.4, 13.7, 6.8, 12.1, 5.3]
})

# Fit logistic regression model
model = smf.logit('rejection ~ age + hla_mismatch + gene_expression', data=data).fit()
print(model.summary())

# Predict rejection probabilities
predicted_probs = model.predict()
print(f"Predicted rejection probabilities: {predicted_probs.values}")
```

**Multiple testing correction & power analysis:**

**Multiple testing problem:** When testing 20,000 genes for differential expression, you expect 1,000 false positives at α = 0.05 even if no genes are truly different.

**Correction methods:**
- **Bonferroni correction:** Divide α by number of tests (conservative)
- **Benjamini-Hochberg:** Controls false discovery rate (preferred for genomics)

```python
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power

# Multiple testing correction
p_values = [0.001, 0.05, 0.02, 0.08, 0.003, 0.12, 0.009, 0.15]
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Power analysis
required_n = ttest_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Required sample size per group for 80% power: {required_n:.0f}")
```

### 2.3. Ethics & Compliance

**IRB processes, informed consent essentials:**

**Institutional Review Board (IRB)** review ensures ethical compliance:
- **Role:** Review research protocols for ethical standards
- **Risk categories:** Minimal risk, greater than minimal risk, high risk
- **Health data considerations:** De-identification requirements, data sharing agreements

**Informed consent elements:**
- **Risks and benefits:** Clear explanation of study procedures and potential outcomes
- **Alternatives:** Other available treatments or procedures
- **Right to withdraw:** Participants can leave the study at any time
- **Special populations:** Additional protections for minors, cognitively impaired

**HIPAA, GDPR, secure data handling:**

**HIPAA Compliance:** Protected health information (PHI) safeguards
- **Technical safeguards:** Encryption, access controls, audit trails
- **Administrative safeguards:** Workforce training, assigned security responsibilities
- **Physical safeguards:** Workstation security, facility access controls

**GDPR Considerations:** European privacy regulations affecting international research
- **Right to erasure:** Participants can request data deletion
- **Data portability:** Participants can request their data in machine-readable format
- **Explicit consent:** Clear, specific consent for data processing

### 2.4. Twofold Research Approach

**Engage subject-matter experts (PIs, literature review):**

Successful health data science requires close collaboration with domain experts:
- **Clinical stakeholders:** Physicians, nurses, researchers who understand the medical context
- **Literature review:** Understanding existing research, identifying gaps
- **Hypothesis formation:** Combining clinical insight with data-driven discoveries

**Exploratory Data Analysis on gene expression transplant dataset:**

EDA helps identify patterns and generate hypotheses:
- **Univariate analysis:** Distribution of gene expression levels, clinical variables
- **Bivariate analysis:** Correlations between genes and clinical outcomes
- **Dimensionality reduction:** PCA to identify major sources of variation
- **Visualization:** Heatmaps, scatter plots, survival curves

**Brainstorm study ideas from SME input and EDA insights:**

Combining expert knowledge with data exploration:
- **Clinical questions:** Which biomarkers predict treatment response?
- **Data-driven insights:** Unexpected patterns requiring clinical interpretation
- **Feasibility assessment:** Sample size requirements, data availability
- **Study design selection:** RCT vs. observational, inclusion/exclusion criteria

## 🛠️ Demo Break 2: EDA & Project Generation Walkthrough

**Objective:** Explore gene expression data and propose research questions

**Steps:**
- Load simulated gene expression dataset for transplant patients
- Compute summary statistics and visualize data distributions
- Perform dimensionality reduction (PCA) to identify major patterns
- Create visualizations showing potential clinical associations
- Draft research project proposals based on exploratory findings

**Success Criteria:**
- Generate informative visualizations of high-dimensional genomic data
- Identify patterns suggesting specific biological hypotheses
- Define two project outlines leveraging experimental design and scaling

---

## 3. End-to-End Health Data Science Project Application

### 3.1. CRISP-DM for Health Data Science

The Cross-Industry Standard Process for Data Mining (CRISP-DM) provides a structured framework adapted for health applications.

**Problem definition & stakeholder analysis:**
- **Clinical stakeholders:** Physicians, nurses, hospital administrators
- **Technical stakeholders:** Data engineers, IT security, compliance officers
- **Patient representatives:** Ethics committees, patient advocacy groups
- **Success metrics:** Clinical impact (mortality reduction, readmission rates)

**Data acquisition & quality assessment:**
- **Health data sources:** EHRs, medical imaging, laboratory data, wearables
- **Quality dimensions:** Completeness, accuracy, consistency, timeliness
- **Privacy compliance:** De-identification verification, consent validation

**Exploratory Data Analysis & hypothesis generation:**
- **Univariate analysis:** Distribution shapes, outliers, missing patterns
- **Bivariate analysis:** Correlations, group comparisons, survival curves
- **Clinical questions:** Which biomarkers predict treatment response?

**Modeling & intervention design:**
- **Model selection:** Balancing performance with interpretability
- **Health-specific considerations:** Class imbalance, regulatory compliance
- **Clinical integration:** API development, alert systems

**Evaluation & iteration:**
- **Performance metrics:** ROC-AUC, precision-recall, fairness metrics
- **Clinical validation:** External validation, temporal validation
- **Error analysis:** When and why does the model fail?

**Communication, documentation, reproducibility, deployment:**
- **Audience-specific reporting:** Clinical, technical, regulatory audiences
- **Version control:** Git workflows, environment management
- **Production monitoring:** Model performance, data drift detection

### 3.2. Illustrative Example: Gene Expression in Transplant Patients

**Walk through CRISP-DM phases using the example dataset:**

**Problem Definition:** Predict transplant rejection risk using gene expression profiles to enable early intervention and improve patient outcomes.

**Data Acquisition:** RNA-seq data from transplant biobank (10,000 patients), clinical outcomes from EHR, immunosuppression records from pharmacy systems.

**EDA Insights:** Gene expression varies by time post-transplant, immune-related pathways show strongest associations with rejection.

**Modeling:** Random forest classifier with SHAP explanations, stratified cross-validation by hospital site.

**Validation:** External validation across multiple transplant centers, performance by organ type and demographics.

**Highlight where compute scaling and experimental design intersect:**
- **Distributed computing:** Parallel processing across hospital databases, hyperparameter search on cluster
- **Experimental design:** Prospective cohort with nested case-control analysis, power calculations for biomarker discovery

### 3.3. Additional Project Examples (Brief)

**Clinical trial A/B pipeline:**
- **Structure:** Power analysis → randomization system → data collection → statistical analysis → reporting
- **Tools:** REDCap integration, mobile apps, automated CONSORT diagrams

**EHR cohort study workflow:**
- **Challenges:** Patient selection, temporal alignment, confounding control
- **Methods:** Propensity score matching, survival analysis, sensitivity analyses

**Imaging ML pipeline:**
**Steps:**
- Use `multiprocessing.Pool` to parallelize a simple analysis across multiple patient files
- Measure speedup compared to sequential processing  
- Write a SLURM job script that could run the same analysis on Wynton
- Identify resource requirements (cores, memory, time) for the job

**Success Criteria:** 
- Observe measurable speedup from parallel processing
- Generate a runnable SLURM script with appropriate resource requests
- Understand when parallelization helps vs. hurts performance

---

## 2. Experimental Design & Analysis in Health Data Science

Rigorous experimental design separates robust health research from wishful thinking. The most sophisticated machine learning model can't overcome fundamental flaws in study design or data collection.

### 2.1. Foundations of Experimental Design

#### 2.1.1. Causal Inference in Health Research

**Correlation vs. Causation** remains the central challenge in health research. Just because coffee drinkers have lower rates of heart disease doesn't mean coffee prevents heart attacks - lifestyle factors like exercise and diet likely confound this relationship.

The **Bradford Hill criteria** provide a framework for assessing causal relationships in observational health data:
- **Strength of association:** Larger effect sizes suggest causation
- **Dose-response relationship:** More exposure leads to greater effect  
- **Temporal sequence:** Cause must precede effect
- **Biological plausibility:** Mechanism makes scientific sense

**Confounding variables** affect both treatment assignment and outcomes, creating spurious associations. In our gene expression transplant study, factors like patient age, kidney function, and immunosuppressive medications could all confound the relationship between gene expression patterns and rejection risk.

**Counterfactual Framework:** What would have happened without the intervention? This fundamental question drives causal inference methods in health research.

**Control strategies** help isolate causal effects:
- **Randomization** balances known and unknown confounders across groups
- **Matching** pairs similar patients receiving different treatments
- **Stratification** analyzes subgroups separately to control for key variables
- **Regression adjustment** statistically controls for measured confounders

#### 2.1.2. Randomized Controlled Trials (RCTs)

**Design Principles:**
- **Randomization:** Eliminates selection bias, balances known/unknown confounders
- **Control Groups:** Placebo, active control, historical control, no treatment
- **Blinding:** Single-blind (participant), double-blind (participant + researcher), triple-blind (+ analyst)

**Randomization Methods:**
- **Simple Randomization:** Coin flip equivalent, may lead to imbalanced groups
- **Block Randomization:** Ensures balanced groups within time periods
- **Stratified Randomization:** Balance within important subgroups (age, sex, severity)
- **Adaptive Randomization:** Adjust probabilities based on interim results

#### 2.1.3. Observational Study Designs

**Cohort Studies:**
- **Prospective:** Follow participants forward in time from exposure to outcome
- **Retrospective:** Look back at historical exposure and outcome data
- **Health Example:** Framingham Heart Study (cardiovascular risk factors)

**Case-Control Studies:**
- **Design:** Compare cases (with disease) to controls (without disease)
- **Advantages:** Efficient for rare diseases, faster than cohort studies
- **Health Example:** Smoking and lung cancer association studies

**Cross-Sectional Studies:**
- **Design:** Snapshot of population at single time point
- **Use Cases:** Disease prevalence, risk factor distribution
- **Limitations:** Cannot establish temporal relationships

#### 2.1.4. Variance Reduction Techniques

**CUPED (Controlled-experiment Using Pre-Experiment Data):**
- **Concept:** Use pre-treatment covariates to reduce outcome variance
- **Formula:** Y_adjusted = Y_post - θ(X_pre - μ_pre)
- **Health Application:** Use baseline biomarkers to improve treatment effect detection

**Blocking and Stratification:**
- **Blocking:** Group similar units before randomization
- **Stratification:** Analyze subgroups separately, then combine
- **Health Example:** Stratify by hospital site, disease severity, or demographics

**Paired Designs:**
- **Matched Pairs:** Each treatment unit matched with similar control unit
- **Crossover Designs:** Each participant receives multiple treatments
- **Health Example:** Twin studies, before/after treatment comparisons

#### 2.1.5. Ethics & Regulatory Considerations

**Informed Consent:**
- **Elements:** Risks, benefits, alternatives, right to withdraw
- **Special Populations:** Minors, cognitively impaired, emergency situations
- **Digital Health:** Data collection, algorithmic decision-making consent

**Institutional Review Board (IRB):**
- **Role:** Review research protocols for ethical compliance
- **Risk Categories:** Minimal risk, greater than minimal risk, high risk
- **Health Data:** De-identification requirements, data sharing agreements

**Data Privacy & Security:**
- **HIPAA Compliance:** Protected health information (PHI) safeguards
- **GDPR Considerations:** Right to erasure, data portability, explicit consent
- **Technical Safeguards:** Encryption, access controls, audit trails

### 2.2. Statistical Methods & Python Tools

#### 2.2.1. Basic Statistical Tests

Choosing appropriate statistical tests depends on your data type, sample size, and research question. Health data often violates assumptions of simple tests, requiring more sophisticated approaches.

**T-Tests:**
- **One-sample:** Compare sample mean to known value
- **Two-sample:** Compare means between groups (unpaired/paired)
- **Assumptions:** Normality, independence, equal variances (for unpaired)
- **Health Example:** Compare blood pressure before/after medication

**Reference Card: `scipy.stats.ttest_ind`**
- **Function:** `scipy.stats.ttest_ind(a, b, equal_var=True)`
- **Purpose:** Compare means between two independent groups
- **Key Parameters:**
    - `a, b`: (Required) Sample data for the two groups being compared
    - `equal_var`: (Optional, default=True) Assume equal population variances. Set to False for Welch's t-test
    - `nan_policy`: (Optional, default='propagate') How to handle NaN values ('propagate', 'raise', 'omit')
    - `alternative`: (Optional, default='two-sided') Alternative hypothesis ('two-sided', 'less', 'greater')

Use t-tests to compare continuous health outcomes between groups:
```python
# Compare blood pressure reduction between treatment and control groups
#FIXME: Implement t-test comparing pre/post treatment measurements
# Include assumption checking and interpretation of results
```

#### 2.2.2. Advanced Statistical Methods

**ANOVA (Analysis of Variance):**
- **One-way:** Compare means across multiple groups
- **Two-way:** Examine two factors and their interaction
- **Repeated Measures:** Account for within-subject correlation
- **Health Example:** Compare treatment effects across multiple drug doses

**Non-parametric Tests:**
- **Mann-Whitney U:** Non-parametric alternative to t-test
- **Kruskal-Wallis:** Non-parametric alternative to ANOVA
- **Wilcoxon Signed-Rank:** Non-parametric paired test
- **Use Cases:** Non-normal data, ordinal outcomes, small samples

**Chi-Square Tests:**
- **Goodness of Fit:** Test if data follows expected distribution
- **Independence:** Test association between categorical variables
- **Health Example:** Test association between smoking status and lung disease

#### 2.2.3. Generalized Linear Models (GLMs)

**Model Components:**
- **Linear Predictor:** β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
- **Link Function:** Connects linear predictor to expected outcome
- **Family Distribution:** Specifies outcome distribution (normal, binomial, Poisson)

**Common GLM Types:**
- **Linear Regression:** Continuous outcomes, identity link, normal distribution
- **Logistic Regression:** Binary outcomes, logit link, binomial distribution
- **Poisson Regression:** Count outcomes, log link, Poisson distribution
- **Cox Proportional Hazards:** Time-to-event outcomes, partial likelihood

**Reference Card: `statsmodels.formula.api`**

```python
#FIXME: Implement logistic regression for disease risk prediction
# Include model fitting, interpretation, and prediction examples
```

#### 2.2.4. Multiple Testing Correction

**The Problem:** Increased Type I error rate when conducting multiple tests. When testing 20,000 genes for differential expression, you expect 1,000 false positives at α = 0.05 even if no genes are truly different.

**Family-Wise Error Rate (FWER):** Probability of making ≥1 Type I error
**False Discovery Rate (FDR):** Expected proportion of false discoveries

**Correction Methods:**
- **FDA guidance:** Special requirements for AI/ML medical devices

### 2.2. Statistical Methods & Python Tools

#### 2.2.1. Basic Statistical Tests

**T-tests** compare means between groups or conditions:
- **One-sample:** Compare sample mean to known population value
- **Two-sample independent:** Compare means between separate groups
- **Paired t-test:** Compare means within same subjects over time
- **Assumptions:** Normality, independence, equal variances (for independent samples)

**Reference Card: `scipy.stats.ttest_ind`**
- **Function:** `scipy.stats.ttest_ind(a, b, equal_var=True)`
- **Purpose:** Perform independent two-sample t-test for group comparisons
- **Key Parameters:**
    - `a, b`: (Required) Array-like sample data for the two groups being compared
    - `equal_var`: (Optional, default=True) Assume equal population variances. Set to False for Welch's t-test
    - `nan_policy`: (Optional, default='propagate') How to handle NaN values ('propagate', 'raise', 'omit')
    - `alternative`: (Optional, default='two-sided') Alternative hypothesis ('two-sided', 'less', 'greater')

```python
import scipy.stats as stats
import numpy as np

# Health example: Compare gene expression levels between rejection/no-rejection groups
rejection_group = [10.5, 15.2, 12.8, 8.9, 14.1, 11.3, 9.7, 13.2]  # Expression levels
no_rejection_group = [6.2, 5.8, 7.3, 4.1, 6.9, 5.4, 4.8, 6.1]      # Expression levels

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(rejection_group, no_rejection_group)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Significant difference: {p_value < 0.05}")

# Check assumptions
print(f"Rejection group normality: {stats.shapiro(rejection_group).pvalue:.3f}")
print(f"No-rejection group normality: {stats.shapiro(no_rejection_group).pvalue:.3f}")
```

#### 2.2.2. Advanced Statistical Methods

**ANOVA (Analysis of Variance)** extends t-tests to multiple groups:
- **One-way ANOVA:** Compare means across multiple independent groups
- **Two-way ANOVA:** Examine two factors and their interaction effects
- **Repeated measures ANOVA:** Account for within-subject correlation over time
- **Health example:** Compare gene expression across multiple immunosuppressive drug regimens

**Non-parametric alternatives** handle assumption violations:
- **Mann-Whitney U:** Non-parametric alternative to independent t-test
- **Kruskal-Wallis:** Non-parametric alternative to one-way ANOVA  
- **Wilcoxon signed-rank:** Non-parametric alternative to paired t-test
- **Use cases:** Non-normal data, ordinal outcomes, small sample sizes

**Chi-square tests** analyze categorical relationships:
- **Goodness of fit:** Test if data follows expected distribution
- **Test of independence:** Examine association between categorical variables
- **Health example:** Test association between gene variants and transplant rejection

#### 2.2.3. Generalized Linear Models (GLMs)

**GLM components** extend linear regression to various outcome types:
- **Linear predictor:** β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
- **Link function:** Connects linear predictor to expected outcome
- **Family distribution:** Specifies outcome distribution (normal, binomial, Poisson)

**Common GLM types** for health research:
- **Linear regression:** Continuous outcomes, identity link, normal distribution
- **Logistic regression:** Binary outcomes, logit link, binomial distribution  
- **Poisson regression:** Count outcomes, log link, Poisson distribution
- **Cox proportional hazards:** Time-to-event outcomes, partial likelihood

**Reference Card: `statsmodels.formula.api`**

```python
import statsmodels.formula.api as smf
import pandas as pd

# Health example: Logistic regression for transplant rejection prediction
data = pd.DataFrame({
    'rejection': [0, 1, 0, 1, 1, 0, 1, 0],
    'age': [45, 62, 38, 58, 67, 41, 55, 49],
    'hla_mismatch': [2, 4, 1, 3, 5, 2, 4, 1],
    'gene_expression': [6.2, 12.8, 5.9, 11.4, 13.7, 6.8, 12.1, 5.3]
})

# Fit logistic regression model
model = smf.logit('rejection ~ age + hla_mismatch + gene_expression', data=data).fit()
print(model.summary())

# Predict rejection probabilities
predicted_probs = model.predict()
print(f"Predicted rejection probabilities: {predicted_probs.values}")

# Odds ratios for clinical interpretation
odds_ratios = np.exp(model.params)
print(f"Odds ratios:\n{odds_ratios}")
```

#### 2.2.4. Multiple Testing Correction

**Multiple testing problem** increases Type I error rates:
- **Issue:** Testing 20,000 genes expects 1,000 false positives at α = 0.05
- **Family-wise error rate (FWER):** Probability of making ≥1 Type I error
- **False discovery rate (FDR):** Expected proportion of false discoveries

**Correction methods** control error rates:
- **Bonferroni correction:** Divide α by number of tests (conservative)
- **Holm-Bonferroni:** Step-down procedure, less conservative than Bonferroni
- **Benjamini-Hochberg:** Controls FDR instead of FWER (preferred for genomics)
- **Permutation tests:** Generate empirical null distribution from data resampling

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Health example: Multiple gene expression associations with rejection
p_values = [0.001, 0.05, 0.02, 0.08, 0.003, 0.12, 0.009, 0.15]
gene_names = ['CXCL9', 'GZMB', 'PDCD1', 'CD3D', 'IFNG', 'IL2RA', 'FOXP3', 'TGFβ1']

# Apply Benjamini-Hochberg FDR correction
rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

# Display results with clinical interpretation
print("Gene Expression Association Results:")
print("-" * 60)
**Goodness of fit tests** determine whether observed data follows an expected distribution, useful for testing Hardy-Weinberg equilibrium in genetic studies.

**Independence tests** examine associations between categorical variables, such as testing relationships between smoking status and lung disease in our transplant cohort.

#### 2.2.3. Generalized Linear Models (GLMs)

**Model Components** provide flexibility for different outcome types:

**Linear predictor** β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ combines predictors linearly.

**Link function** connects the linear predictor to the expected outcome, accommodating non-linear relationships.

**Family distribution** specifies the outcome distribution (normal, binomial, Poisson) appropriate for the data type.

**Common GLM Types** handle diverse health outcomes:

**Linear regression** uses identity link and normal distribution for continuous outcomes like biomarker levels.

**Logistic regression** employs logit link and binomial distribution for binary outcomes like disease presence/absence.

**Poisson regression** uses log link and Poisson distribution for count outcomes like number of hospitalizations.

**Cox proportional hazards models** handle time-to-event outcomes like transplant rejection time using partial likelihood methods.

**Reference Card: `statsmodels.formula.api`**

```python
import statsmodels.formula.api as smf
import pandas as pd

# Health example: Logistic regression for transplant rejection risk
transplant_data = pd.DataFrame({
    'rejection': [0, 1, 0, 1, 1, 0, 1, 0],
    'age': [45, 62, 38, 58, 67, 41, 55, 49],
    'hla_mismatch': [2, 4, 1, 3, 5, 2, 4, 1],
    'gene_expression_score': [0.2, 0.8, 0.1, 0.7, 0.9, 0.3, 0.6, 0.2]
})

# Fit logistic regression model
model = smf.logit('rejection ~ age + hla_mismatch + gene_expression_score', 
                  data=transplant_data).fit()
print(model.summary())

# Predict rejection probabilities
rejection_prob = model.predict()
print(f"Predicted rejection probabilities: {rejection_prob}")

#FIXME: Add model diagnostics, confidence intervals, and clinical interpretation
```

#### 2.2.4. Multiple Testing Correction

**The Problem** arises from increased Type I error rates when conducting multiple statistical tests simultaneously. In genomics research, testing 20,000 genes for differential expression at α = 0.05 would produce 1,000 false positives even if no genes are truly different between groups.

**Family-Wise Error Rate (FWER)** represents the probability of making one or more Type I errors across all tests performed.

**False Discovery Rate (FDR)** measures the expected proportion of false discoveries among all rejected hypotheses, often more appropriate for exploratory research.

**Correction Methods** control error rates through different approaches:

**Bonferroni correction** divides α by the number of tests, providing strong FWER control but potentially excessive conservatism.

**Holm-Bonferroni** uses a step-down procedure that's uniformly more powerful than Bonferroni while maintaining FWER control.

**Benjamini-Hochberg** controls FDR instead of FWER, allowing more discoveries while maintaining an acceptable proportion of false positives.

**Permutation tests** generate empirical null distributions through data resampling, accounting for correlation structure among tests.

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Health example: Multiple biomarker associations in transplant study
p_values = [0.001, 0.05, 0.02, 0.08, 0.003, 0.12, 0.009, 0.15]
biomarkers = ['IL-6', 'TNF-α', 'CRP', 'CXCL10', 'CCL2', 'IFN-γ', 'IL-10', 'TGF-β']

# Apply Benjamini-Hochberg FDR correction
rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

# Display results with clinical interpretation
print("Biomarker Association Results:")
print("-" * 50)
for biomarker, p_orig, p_corr, significant in zip(biomarkers, p_values, p_corrected, rejected):
    status = "SIGNIFICANT" if significant else "Not significant"
    print(f"{biomarker}: p={p_orig:.3f}, p_corrected={p_corr:.3f}, {status}")
```

**Checkpoint:** In our gene expression transplant study, why would we prefer FDR control over FWER control when testing thousands of genes?

#### 2.2.5. Power Analysis & Sample Size Calculation

**Statistical power** represents the probability of detecting a true effect when it exists (1 - Type II error rate). Adequate power prevents wasted resources on underpowered studies that cannot detect clinically meaningful effects.

**Factors affecting power:**
- **Effect size:** Larger effects are easier to detect with smaller samples
- **Sample size:** More data increases power but with diminishing returns
- **Significance level:** Lower α reduces power but decreases false positive risk
- **Variability:** Less noise in measurements increases power to detect signal

**Reference Card: `statsmodels.stats.power`**

```python
from statsmodels.stats.power import ttest_power, tt_solve_power
import matplotlib.pyplot as plt
import numpy as np

# Health example: Power analysis for gene expression biomarker study
effect_sizes = np.arange(0.2, 2.0, 0.1)
sample_sizes = [25, 50, 100, 200]

# Create power curves for different sample sizes
plt.figure(figsize=(12, 8))
for n in sample_sizes:
    powers = [ttest_power(effect_size=es, nobs=n, alpha=0.05) for es in effect_sizes]
    plt.plot(effect_sizes, powers, 'o-', linewidth=2, label=f'n = {n} per group')

plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Power Threshold')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Statistical Power')
plt.title('Power Analysis for Gene Expression Biomarker Study')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate required sample size for specific power
required_n = tt_solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Required sample size per group for 80% power (d=0.5): {required_n:.0f}")

# Power for our transplant study
study_power = ttest_power(effect_size=0.6, nobs=150, alpha=0.05)
print(f"Power with 150 patients per group (d=0.6): {study_power:.3f}")
```

**Clinical considerations** in power analysis:
- **Minimal clinically important difference:** What effect size matters to patients?
- **Recruitment feasibility:** Can we realistically achieve the required sample size?
- **Cost-effectiveness:** Balance statistical power against study costs
- **Ethical considerations:** Minimize patient exposure while maintaining scientific rigor

### 2.3. Health Data Applications

#### 2.3.1. Clinical Trial Design & Analysis

**Phase I Trials:**
- **Objective:** Determine safe dosage ranges and identify dose-limiting toxicities
- **Design:** Dose escalation studies with small sample sizes (20-100 participants)
- **Analysis:** Safety monitoring, toxicity assessment, pharmacokinetic modeling

**Phase II Trials:**
- **Objective:** Assess treatment effectiveness while monitoring safety
- **Design:** Single-arm or randomized studies (100-300 participants)
- **Analysis:** Response rates, progression-free survival, biomarker correlations

**Phase III Trials:**
- **Objective:** Compare new treatment to standard of care
- **Design:** Large randomized controlled trials (300-3,000 participants)
- **Analysis:** Primary/secondary endpoints, interim analyses, subgroup analyses

**Adaptive trial designs** allow modifications based on accumulating data:
- **Sample size re-estimation** maintains power as effect size estimates evolve
- **Biomarker-guided randomization** enriches for patients likely to respond
- **Seamless phase transitions** move promising treatments efficiently through development

#### 2.3.2. Electronic Health Record (EHR) Studies

**Observational study challenges** in EHR research:
- **Selection bias:** Who gets recorded in the EHR and why?
- **Information bias:** Missing data, measurement errors, coding inconsistencies
- **Confounding:** Indication bias, unmeasured confounders, time-varying effects

**Statistical approaches** for causal inference from EHR data:
- **Propensity score methods:** Balance treatment groups on observed covariates
- **Instrumental variables:** Leverage natural experiments for causal estimation
- **Difference-in-differences:** Compare changes over time between groups
- **Regression discontinuity:** Exploit arbitrary treatment assignment thresholds

For our transplant study, EHR data could provide long-term outcomes, but we'd need careful consideration of loss to follow-up and hospital-specific recording practices.

#### 2.3.3. Genomics & High-Dimensional Data

**Multiple testing challenges** in genomics research:
- **GWAS studies:** Test millions of SNPs simultaneously across the genome
- **RNA-seq analysis:** Test thousands of genes for differential expression
- **Proteomics:** Analyze hundreds of proteins across different conditions

**Specialized methods** for high-dimensional health data:
- **Gene set enrichment analysis:** Test groups of related genes instead of individual markers
- **Pathway analysis:** Incorporate biological knowledge into statistical testing
- **Machine learning approaches:** Use prediction performance as selection criterion

**Validation strategies** ensure reproducible genomics findings:
- **Replication cohorts:** Test findings in independent populations
- **Functional validation:** Confirm biological mechanisms in laboratory studies
- **Clinical validation:** Demonstrate utility for patient care decisions

## 🛠️ Demo Break 2: EDA & Project Generation Walkthrough

**Objective:** Explore gene expression data and propose research questions

**Steps:**
- Load simulated gene expression dataset for transplant patients
- Compute summary statistics and visualize data distributions
- Perform dimensionality reduction (PCA) to identify major patterns
- Create visualizations showing potential clinical associations
- Draft research project proposals based on exploratory findings

**Success Criteria:**
- Generate informative visualizations of high-dimensional genomic data
- Identify patterns suggesting specific biological hypotheses
- Propose feasible research questions combining statistical rigor with clinical relevance

---

## 3. End-to-End Health Data Science Project Application

### 3.1. CRISP-DM for Health Data Science

The Cross-Industry Standard Process for Data Mining (CRISP-DM) provides a structured framework that prevents common pitfalls in health data science projects. Adapted for health applications, it emphasizes clinical relevance and regulatory compliance at each stage.

#### 3.1.1. Phase 1: Problem Definition & Scoping

**Stakeholder Identification & Requirements:**

Health data science projects involve diverse stakeholders with different priorities and constraints. **Clinical stakeholders** including physicians, nurses, researchers, and hospital administrators focus on patient outcomes and operational efficiency. **Technical stakeholders** such as data engineers, ML engineers, IT security, and compliance officers ensure technical feasibility and regulatory adherence. **Patient representatives** and ethics committees advocate for patient interests and safety.

**Business Understanding** requires translating clinical problems into data science questions:
- **Clinical problem definition:** What specific medical question are we addressing?
- **Success metrics:** How will we measure clinical impact (mortality reduction, readmission rates, quality of life)?
- **Constraints:** Budget limitations, timeline requirements, regulatory approval processes
- **Risk assessment:** Patient safety considerations, data privacy risks, algorithm bias potential

**Experimental Design Integration** happens at the project outset:
- **Study type selection:** RCT vs. observational cohort vs. case-control design
- **Power analysis:** Sample size calculations based on expected effect sizes
- **Randomization strategy:** Simple, block, stratified, or adaptive randomization
- **Endpoint specification:** Primary outcomes, secondary measures, safety endpoints

For our gene expression transplant study: Clinical stakeholders want early rejection prediction; success means 20% reduction in missed rejections; constraints include IRB approval timeline; experimental design uses prospective cohort with nested case-control analysis.

#### 3.1.2. Phase 2: Data Acquisition & Understanding

**Health Data Sources** span multiple systems and formats:
- **Electronic Health Records:** Epic, Cerner, Allscripts extracts with ICD codes, medications, lab values
- **Medical Imaging:** DICOM files from PACS systems, pathology slides, radiology reports
- **Laboratory Data:** Blood tests, genetic sequencing results, biomarker panels
- **Wearable/IoT Devices:** Continuous glucose monitors, heart rate sensors, activity trackers
- **Research Datasets:** PhysioNet, MIMIC-IV, UK Biobank, All of Us Research Program

**Data Quality Assessment** identifies potential issues early:
- **Completeness:** Missing data patterns, dropout rates, loss to follow-up
- **Accuracy:** Validation rules, outlier detection, cross-source verification
- **Consistency:** Standardized terminologies (ICD-10, SNOMED), unit conversions
- **Timeliness:** Data freshness, lag between events and recording
- **Privacy compliance:** De-identification verification, PHI detection, consent validation

**Technical Infrastructure** considerations:
- **Data storage:** HIPAA-compliant cloud services, on-premise secure servers
- **Access controls:** Role-based permissions, audit logging, VPN requirements
- **Data formats:** HL7 FHIR for interoperability, DICOM for imaging, Parquet for efficient analytics
- **ETL pipelines:** Apache Airflow or Prefect for automated data processing workflows

Our transplant study sources RNA-seq data from biobank, clinical outcomes from EHR, and immunosuppression records from pharmacy systems, requiring careful coordination across data silos.

#### 3.1.3. Phase 3: Exploratory Data Analysis (EDA)

**Univariate Analysis** examines individual variables:
- **Continuous variables:** Distribution shapes, outliers, normality testing
- **Categorical variables:** Frequency tables, missing data patterns
- **Temporal patterns:** Seasonality effects, trends over time, survival curves
- **Health-specific metrics:** Reference ranges, clinical significance thresholds

**Bivariate/Multivariate Analysis** reveals relationships:
- **Correlation analysis:** Feature relationships, multicollinearity detection
- **Group comparisons:** Treatment vs. control, diseased vs. healthy populations
- **Survival analysis:** Kaplan-Meier curves, log-rank tests for time-to-event outcomes
- **Dimensionality reduction:** PCA for genomics data, t-SNE for clustering visualization

**Hypothesis Generation** combines domain knowledge with data patterns:
- **Clinical questions:** Which biomarkers predict treatment response?
- **Data-driven insights:** Unexpected patterns requiring clinical interpretation
- **Confounding assessment:** Early identification of potential confounders
- **Effect size estimation:** Preliminary estimates for sample size validation

In our transplant EDA, we might discover that gene expression varies significantly by time post-transplant, leading to stratified analyses and time-adjusted models.

#### 3.1.4. Phase 4: Data Preparation & Feature Engineering

**Missing Data Handling** requires understanding missingness mechanisms:
- **Missing Completely at Random (MCAR):** Missingness unrelated to any variables
- **Missing at Random (MAR):** Missingness depends on observed variables
- **Missing Not at Random (MNAR):** Missingness depends on unobserved factors
- **Imputation strategies:** Mean/median for simple cases, KNN or multiple imputation for complex patterns
- **Health-specific considerations:** Lab values below detection limits, censored survival times

**Feature Engineering** creates clinically meaningful variables:
- **Temporal features:** Time since diagnosis, treatment duration, age at onset
- **Clinical risk scores:** APACHE II, SOFA score, Framingham Risk Score
- **Derived biomarkers:** Ratios (LDL/HDL), differences (systolic-diastolic), rates of change
- **Text features:** NLP on clinical notes, sentiment analysis, named entity recognition

**Data Transformation** prepares features for modeling:
- **Normalization:** Min-max scaling, z-score standardization, robust scaling for outliers
- **Encoding:** One-hot encoding for categories, target encoding for high cardinality
- **Outlier treatment:** Clinical vs. statistical outliers requiring domain expertise
- **Distribution transformation:** Log transformation for skewed biomarkers, Box-Cox transforms

For our transplant study, we engineer features like "days since transplant," "immunosuppression burden score," and "donor-recipient HLA mismatch index."

#### 3.1.5. Phase 5: Modeling & Intervention Design

**Model Selection Strategy** balances performance with interpretability:
- **Baseline models:** Logistic regression, linear regression for interpretable benchmarks
- **Tree-based models:** Random forests for tabular health data, XGBoost for competitions
- **Deep learning:** CNNs for medical imaging, RNNs for time series, transformers for text
- **Ensemble methods:** Voting classifiers, stacking, blending multiple model types

**Health-Specific Considerations:**
- **Interpretability requirements:** SHAP values, LIME, feature importance for clinical trust
- **Class imbalance:** SMOTE, cost-sensitive learning, focal loss for rare diseases
- **Regulatory compliance:** FDA guidance for AI/ML devices, bias testing, fairness metrics
- **Clinical integration:** API development, real-time scoring, alert system design

**Experimental Setup** ensures robust evaluation:
- **Train/validation/test splits:** Temporal splits for time series, patient-level splits for longitudinal data
- **Cross-validation:** Stratified CV for imbalanced classes, group CV for clustered data
- **Hyperparameter optimization:** Grid search, random search, Bayesian optimization
- **Early stopping:** Prevent overfitting, monitor validation metrics

Our transplant rejection model uses random forest for interpretability, stratified CV by hospital site, and SHAP for feature importance explanation to clinicians.

#### 3.1.6. Phase 6: Evaluation & Iteration

**Performance Metrics** align with clinical priorities:
- **Classification:** ROC-AUC, precision-recall curves, sensitivity/specificity at clinical thresholds
- **Regression:** RMSE, MAE, R-squared, clinical significance of prediction errors
- **Survival analysis:** C-index, Brier score, time-dependent ROC curves
- **Fairness metrics:** Equalized odds, demographic parity, individual fairness across subgroups

**Clinical Validation** tests real-world applicability:
- **External validation:** Performance on different hospital systems, patient populations
- **Temporal validation:** Model performance over time, concept drift detection
- **Subgroup analysis:** Performance by age, sex, race, comorbidities
- **Clinical expert review:** Sanity checks, domain knowledge validation

**Error Analysis** identifies improvement opportunities:
- **Failure mode analysis:** When and why does the model fail?
- **Calibration assessment:** Are predicted probabilities well-calibrated?
- **Feature attribution:** Which features drive specific predictions?
- **Bias detection:** Systematic errors across demographic groups

For our transplant model, we prioritize sensitivity at 95% specificity (missing rejection is dangerous), validate across multiple transplant centers, and analyze performance by organ type and patient demographics.

#### 3.1.7. Phase 7: Communication & Reporting

**Audience-Specific Reporting** tailors content to stakeholder needs:
- **Clinical audiences:** Focus on clinical significance, patient outcomes, workflow integration
- **Technical audiences:** Model architecture, performance metrics, computational requirements
- **Regulatory audiences:** Validation studies, bias testing, safety considerations
- **Executive audiences:** Return on investment, operational impact, strategic implications

**Visualization Strategy** enhances understanding:
- **Clinical dashboards:** Real-time patient monitoring, risk stratification displays
- **Research reports:** Publication-quality figures, statistical test results
- **Interactive tools:** Altair charts for exploration, Dash apps for stakeholder engagement
- **Automated reporting:** MkDocs for documentation, GitHub Pages for sharing

Our transplant study creates a clinical decision support dashboard showing rejection risk scores, gene expression heatmaps, and model explanation for each patient.

#### 3.1.8. Phase 8: Version Control & Reproducibility

**Code Management** ensures collaborative development:
- **Git workflows:** Feature branches, pull requests, code review processes
- **Environment management:** Docker containers, conda environments, requirements.txt
- **Data versioning:** DVC, MLflow for data and model versioning
- **Experiment tracking:** Weights & Biases, MLflow, Neptune for experiment management

**Documentation Standards** enable knowledge transfer:
- **Code documentation:** Docstrings, type hints, comprehensive README files
- **Analysis documentation:** Jupyter notebooks with narrative, methodology descriptions
- **Model documentation:** Model cards, bias testing reports, performance summaries
- **Protocol documentation:** IRB protocols, statistical analysis plans, data sharing agreements

#### 3.1.9. Phase 9: Deployment & Monitoring

**Deployment Strategies** match clinical workflow needs:
- **Batch scoring:** Scheduled model runs for population risk stratification
- **Real-time inference:** API endpoints for clinical decision support, immediate alerts
- **Edge deployment:** On-device models for wearables, point-of-care testing
- **Federated learning:** Models trained across institutions without data sharing

**Production Monitoring** maintains model performance:
- **Performance monitoring:** Model accuracy over time, prediction distribution shifts
- **Data drift detection:** Changes in feature distributions, new data patterns
- **Fairness monitoring:** Ongoing bias detection, equitable performance tracking
- **Clinical outcome tracking:** Patient outcomes, intervention effectiveness, safety signals

![CRISP-DM health data science workflow](#FIXME)

Each phase incorporates both distributed computing and experimental design considerations. Data acquisition might require parallel processing of multiple hospital databases. Modeling could involve distributed hyperparameter search across cluster nodes. Evaluation demands careful experimental design to ensure fair comparison across patient subgroups.

### 3.2. Health Data Project Examples by Type

#### 3.2.1. Clinical Trial A/B Test Pipeline

**Project Structure** optimizes trial efficiency:
- **Power analysis phase:** Sample size calculation, effect size estimation, interim analysis planning
- **Randomization system:** Web-based randomization with stratification factors
- **Data collection:** REDCap integration, mobile app data capture, EHR integration
- **Statistical analysis:** Intention-to-treat analysis, per-protocol analysis, subgroup analyses
- **Reporting:** CONSORT diagram, primary endpoint analysis, safety reporting

#### 3.2.2. EHR Cohort Study Workflow

**Data extraction** requires careful patient selection:
- **Patient identification:** ICD codes, medication lists, procedure codes
- **Temporal alignment:** Index dates, follow-up periods, censoring rules
- **Covariate collection:** Demographics, comorbidities, baseline characteristics
- **Outcome ascertainment:** Primary endpoints, time-to-event outcomes, composite endpoints

**Analysis pipeline** addresses observational study challenges:
- **Propensity score development:** Logistic regression for treatment assignment prediction
- **Matching/stratification:** 1:1 matching, inverse probability weighting
- **Survival analysis:** Cox proportional hazards, competing risks models
- **Sensitivity analyses:** Unmeasured confounding, missing data assumptions

#### 3.2.3. Medical Imaging ML Workflow

**Image processing pipeline** handles large-scale medical images:
- **DICOM handling:** Anonymization, format conversion, metadata extraction
- **Preprocessing:** Normalization, windowing, artifact removal
- **Augmentation:** Rotation, scaling, elastic deformations for training data
- **Quality assurance:** Automated quality checks, outlier detection

**Model development** leverages distributed computing:
- **Architecture selection:** ResNet, DenseNet, Vision Transformers
- **Transfer learning:** ImageNet pretraining, domain adaptation
- **Distributed training:** Multi-GPU training, gradient accumulation
- **Evaluation:** ROC analysis, CAM visualization, radiologist agreement studies

