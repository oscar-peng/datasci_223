# Lecture 10: Experimentation, Research Design, and Distributed Computing in Health Data Science

## Course Recap & Integration

**Course Summary:**

1. Dev Environment Setup & Python Fundamentals (Jupyter, VSCode, Git, Python basics)
2. Handling Larger-than-Memory Data with Polars (efficient tabular data, chunking, out-of-core)
3. SQL for Data Analysis (joins, aggregations, health data queries)
4. Regression Concepts & Time-Series Forecasting (supervised learning, time-based splits, ARIMA, feature engineering)
5. Introduction to ML & Classification (classification models, evaluation metrics, hyperparameter tuning, model selection)
6. Neural Network Flavors & Applications (PyTorch, intro to deep learning)
7. Deep Learning: LLM APIs & Transformers (NLP, clinical text, prompt engineering)
8. Computer Vision: Identifying and Tracking (medical imaging, PyTorch, segmentation)
9. Data Visualization, Diagramming, Reporting & Dashboards (Altair, Dash, MkDocs, reproducible reporting)
10. Experimentation, Research Design, and Distributed Computing in Health Data Science

**Real-World Integration Stories:**
    - **AI-Powered Clinical Decision Support System:** SQL data extraction from EHRs + deep learning risk prediction + interactive Dash dashboards for clinicians + distributed training on Wynton
    - **Large-Scale Genomics Study:** Distributed variant calling on 10,000+ samples + experimental design for population stratification + visualization pipelines for publication + automated reporting
    - **Real-Time Pandemic Monitoring:** Big data processing of streaming health metrics + computer vision for symptom detection + live dashboards for public health officials + version-controlled analysis pipelines
    - **Personalized Medicine Pipeline:** Multi-modal data integration (genomics + imaging + EHR) + ensemble ML approaches + A/B testing for treatment recommendations + deployment monitoring

## 0. Introduction & Review

- **Purpose:** Connect prior skills to experimental design and distributed computing in health data science.
- **Review of Key Demos:**
    - Mermaid diagramming for workflow visualization
    - Altair for interactive health data visualizations
    - MkDocs for automated reporting
    - Dash for building interactive dashboards
- **Transition:** From visualization/reporting to experimental methodology and distributed workflows

## 1. Distributed Computing & Scaling in Health Data Science

### 1.1. Why Distributed Computing?

- **Motivation:** Health datasets (genomics, imaging, EHR) often exceed single-machine capacity.
- **Scale Examples in Health Data:**
    - **Genomics:** UK Biobank (500K+ genomes, 100TB+ data)
    - **Imaging:** RadImageNet (2M+ medical images)
    - **EHR:** Epic Cosmos (180M+ patient records)
    - **Wearables:** Apple Health Study (400K+ participants, continuous streaming)

#### 1.1.1. When to Scale - Specific Health Use Cases

- **Large Parameter Sweeps:**
    - Hyperparameter tuning for drug discovery models (10K+ combinations)
    - Neural architecture search for medical image classification
    - Ensemble model optimization across multiple hospitals
- **Cross-Validation on Big Cohorts:**
    - 10-fold CV on 1M+ patient EHR records
    - Leave-one-site-out validation across hospital networks
    - Temporal validation on longitudinal health data
- **Monte Carlo Simulations:**
    - Clinical trial power analysis (100K+ simulations)
    - Epidemiological modeling for disease spread
    - Health economic modeling with uncertainty quantification
- **Genomics Pipelines:**
    - Variant calling on whole genome sequences (30x coverage, 3GB per sample)
    - RNA-seq differential expression (50K+ genes × 1000+ samples)
    - GWAS meta-analysis across multiple cohorts
- **Medical Image Analysis:**
    - Radiomics feature extraction from 10K+ CT scans
    - Pathology slide analysis (gigapixel images, 1000+ slides)
    - Real-time MRI reconstruction during surgery

#### 1.1.2. Scale Thresholds & Decision Points

- **Memory:** > 32GB RAM needed → consider distributed approaches
- **Compute Time:** > 4 hours on single core → parallelize
- **Data Size:** > 100GB datasets → out-of-core or distributed processing
- **Model Complexity:** > 1B parameters → model parallelism required

### 1.2. Core Concepts & Health Data Context

#### 1.2.1. Threads vs. Processes in Health Computing

- **Threads (Shared Memory):**
    - **Use Case:** I/O-bound tasks like downloading DICOM images
    - **Health Example:** Concurrent API calls to FHIR servers
    - **Python:** `threading.Thread`, `concurrent.futures.ThreadPoolExecutor`
- **Processes (Isolated Memory):**
    - **Use Case:** CPU-intensive tasks like image processing
    - **Health Example:** Parallel CNNs training on medical images
    - **Python:** `multiprocessing.Process`, `concurrent.futures.ProcessPoolExecutor`

#### 1.2.2. Parallelism vs. Concurrency

- **Parallelism (True Simultaneous):**
    - Multiple CPU cores processing different patients simultaneously
    - Example: 8-core machine analyzing 8 patient records in parallel
- **Concurrency (Interleaved Tasks):**
    - Single core switching between downloading and processing tasks
    - Example: Download next patient data while processing current patient

#### 1.2.3. CPU-bound vs. I/O-bound Optimization

- **CPU-bound Health Tasks:**
    - Deep learning model training
    - Genomic sequence alignment
    - Statistical model fitting on large datasets
    - **Strategy:** Use process-based parallelism, scale to more cores
- **I/O-bound Health Tasks:**
    - Downloading medical images from PACS
    - Querying multiple hospital databases
    - Reading large files from network storage
    - **Strategy:** Use thread-based concurrency, increase I/O throughput

### 1.3. Architectures & Evolution in Health Computing

#### 1.3.1. Single-Machine Scaling

- **Multi-threading:**
    - **Health Use:** Concurrent DICOM downloads, database queries
    - **Python Tools:** `threading`, `asyncio`
    - **Limitations:** GIL (Global Interpreter Lock) limits CPU parallelism
- **Multi-processing:**
    - **Health Use:** Parallel image processing, model training
    - **Python Tools:** `multiprocessing`, `joblib`
    - **Resource Planning:** 1-2 processes per CPU core for compute tasks

#### 1.3.2. Cluster Computing Evolution

- **Traditional HPC (MPI, Beowulf):**
    - **Health Use:** Large-scale genomics, climate health modeling
    - **Characteristics:** Tight coupling, low-latency networking
- **Grid Computing (SGE/SLURM):**
    - **Health Use:** Embarrassingly parallel health data analysis
    - **UCSF Wynton:** 4,000+ cores for health research
- **Modern Distributed (Kubernetes, Spark):**
    - **Health Use:** Real-time health monitoring, elastic ML pipelines
    - **Benefits:** Auto-scaling, fault tolerance, cloud integration

#### 1.3.3. Cloud Computing Patterns

- **Elastic Scaling:**
    - **Health Example:** Scale up during flu season for symptom tracking
    - **Auto-scaling:** Based on queue length or resource utilization
- **Containers (Docker/Kubernetes):**
    - **Health Use:** Reproducible analysis environments across hospitals
    - **Benefits:** Environment consistency, easy deployment
- **Serverless Computing:**
    - **Health Example:** Trigger analysis when new patient data arrives
    - **AWS Lambda/Azure Functions:** Event-driven health data processing

#### 1.3.4. Distributed Computing Patterns for Health

- **Map-Reduce Pattern:**
    - **Health Example:** Process 1M patient records independently, then aggregate statistics
    - **Map:** Extract features from each patient record
    - **Reduce:** Aggregate population-level statistics
- **Orchestrator/Worker Pattern:**
    - **Health Example:** Hyperparameter search for drug discovery models
    - **Orchestrator:** Manages parameter combinations and results
    - **Workers:** Train models with specific parameter sets
- **Pipeline Pattern:**
    - **Health Example:** Medical image analysis workflow
    - **Stage 1:** DICOM preprocessing (anonymization, format conversion)
    - **Stage 2:** Feature extraction (radiomics, deep features)
    - **Stage 3:** Classification (tumor detection, diagnosis prediction)

### 1.4. Python Approaches & Tools

#### 1.4.1. Local Parallelism Libraries

- **multiprocessing Module:**
    - **Process Pool:** `multiprocessing.Pool(processes=4)`
    - **Shared Memory:** `multiprocessing.Manager()` for large datasets
    - **Health Example:** Parallel patient cohort analysis
- **concurrent.futures:**
    - **Thread Pool:** `ThreadPoolExecutor(max_workers=8)` for I/O tasks
    - **Process Pool:** `ProcessPoolExecutor(max_workers=4)` for CPU tasks
    - **Health Example:** Concurrent medical database queries
- **joblib:**
    - **Parallel:** `Parallel(n_jobs=4)(delayed(func)(arg) for arg in args)`
    - **Memory Mapping:** Efficient sharing of large NumPy arrays
    - **Health Example:** Parallel scikit-learn model training

#### 1.4.2. Distributed Computing Frameworks

- **Dask:**
    - **Use Case:** Pandas/NumPy operations on larger-than-memory health datasets
    - **Features:** Lazy evaluation, familiar APIs, task graphs
    - **Health Example:** EHR analysis on 100GB+ datasets
- **Ray:**
    - **Use Case:** Distributed ML training, hyperparameter tuning
    - **Features:** Actor model, distributed objects, auto-scaling
    - **Health Example:** Distributed deep learning for medical imaging
- **PySpark:**
    - **Use Case:** Big data processing, ETL pipelines
    - **Features:** Distributed DataFrames, MLlib, streaming
    - **Health Example:** Hospital data warehouse processing

#### 1.4.3. Reference Card: multiprocessing.Pool

**Function:** `multiprocessing.Pool(processes=None)`
**Purpose:** Create a pool of worker processes for parallel execution
**Key Parameters:**

- `processes`: (Optional, default=None) Number of worker processes. If None, uses number of CPU cores
- `initializer`: (Optional, default=None) Function to run when each worker starts
- `initargs`: (Optional, default=()) Arguments for initializer function
- `maxtasksperchild`: (Optional, default=None) Maximum tasks per worker before restart

**Key Methods:**

- `map(func, iterable)`: Apply function to each item in parallel
- `starmap(func, iterable)`: Like map but unpacks arguments from tuples
- `apply_async(func, args)`: Asynchronous function application
- `close()`: Prevent more tasks from being submitted
- `join()`: Wait for worker processes to exit

**Health Data Example:**

```python
import multiprocessing as mp
import pandas as pd
import time

def analyze_patient_cohort(patient_file):
    """Analyze a single patient cohort file"""
    df = pd.read_csv(patient_file)
    # Simulate complex analysis
    time.sleep(0.1)
    return {
        'file': patient_file,
        'n_patients': len(df),
        'avg_age': df['age'].mean(),
        'high_risk_count': (df['risk_score'] > 0.8).sum()
    }

# Sequential processing
start_time = time.time()
patient_files = [f'cohort_{i}.csv' for i in range(20)]
sequential_results = [analyze_patient_cohort(f) for f in patient_files]
sequential_time = time.time() - start_time

# Parallel processing
start_time = time.time()
with mp.Pool(processes=4) as pool:
    parallel_results = pool.map(analyze_patient_cohort, patient_files)
parallel_time = time.time() - start_time

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

### 1.5. UCSF Wynton SGE/SLURM & Resource Guidance

#### 1.5.1. Wynton Cluster Overview

- **Hardware Specifications:**
    - **CPU Cores:** 4,000+ Intel/AMD cores across 200+ nodes
    - **GPUs:** 88 high-end GPUs (H200, H100, L40S, V100, RTX series)
    - **Memory:** Nodes ranging from 64GB to 1.5TB RAM
    - **Storage:** 2PB shared storage, high-speed Lustre filesystem
    - **Network:** 100GbE and InfiniBand for low-latency communication

#### 1.5.2. SGE vs SLURM

- **SGE (Sun Grid Engine) - Wynton:**
    - **Status:** Current system, being superseded by CoreHPC & SLURM
    - **Commands:** `qsub`, `qstat`, `qdel`, `qhold`
    - **Script Format:** `#$ -pe smp 4` for parallel environment
- **SLURM (Simple Linux Utility for Resource Management) - CoreHPC:**
    - **Status:** Primary scheduler, recommended for new projects
    - **Commands:** `sbatch`, `squeue`, `scancel`, `scontrol`
    - **Script Format:** `#SBATCH --cpus-per-task=4` for resource requests

#### 1.5.3. Health Data Job Submission Examples

**SGE Example - Genomics Variant Calling:**

```bash
#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l mem_free=32G
#$ -l h_rt=24:00:00
#$ -N variant_calling_chr1
#$ -o logs/variant_calling_chr1.out
#$ -e logs/variant_calling_chr1.err

module load gatk/4.3.0
module load samtools/1.15

# Process chromosome 1 variants
gatk HaplotypeCaller \
    -R reference/hg38.fa \
    -I samples/patient_001.bam \
    -O variants/patient_001_chr1.vcf \
    -L chr1 \
    --native-pair-hmm-threads 8
```

**SLURM Example - Deep Learning Medical Image Classification:**

```bash
#!/bin/bash
#SBATCH --job-name=medical_cnn_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cnn_training_%j.out
#SBATCH --error=logs/cnn_training_%j.err

module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Train CNN on chest X-ray dataset
python train_medical_cnn.py \
    --data_dir /wynton/group/health/datasets/chest_xray/ \
    --model resnet50 \
    --batch_size 32 \
    --epochs 100 \
    --gpus 2 \
    --output_dir models/chest_xray_resnet50/
```

#### 1.5.4. Resource Estimation Guidelines

**CPU Requirements by Task:**

- **Data Preprocessing:** 4-8 cores, moderate I/O
- **Statistical Analysis:** 1-16 cores, depends on method complexity
- **Machine Learning Training:** 8-32 cores, high CPU utilization
- **Genomics Pipelines:** 16-64 cores, memory and I/O intensive

**Memory Requirements by Data Type:**

- **Tabular Health Data:** 2-4x dataset size in RAM
- **Medical Images:** 8-16GB for CNN training, 32GB+ for large models
- **Genomics:** 16-128GB depending on reference genome and sample size
- **Time Series:** 4-8x dataset size for complex feature engineering

**GPU Requirements:**

- **Deep Learning Training:** 1-4 GPUs depending on model size
- **Inference:** Often 1 GPU sufficient, batch processing
- **Computer Vision:** 2-8 GPUs for large medical image datasets
- **NLP/LLMs:** 4-8 high-memory GPUs (H100, A100) for large models

#### 1.5.5. Health Data Use Cases & Resource Specs

**Distributed Variant Calling (WGS Analysis):**

```bash
# Resource needs: 16 cores, 64GB RAM, 24 hours per sample
#SBATCH --array=1-1000  # Process 1000 samples
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

**Parallel Medical Image Processing:**

```bash
# Resource needs: 8 cores, 32GB RAM, GPU optional
#SBATCH --array=1-5000  # Process 5000 medical images
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
```

**Large-Scale EHR Feature Engineering:**

```bash
# Resource needs: 32 cores, 128GB RAM, high I/O
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --constraint=ssd  # Use SSD storage for faster I/O
```

#### 1.5.6. Job Monitoring & Optimization

**Monitoring Commands:**

```bash
# Check job status
squeue -u $USER                    # Your jobs
squeue -j 12345                    # Specific job
scontrol show job 12345            # Detailed job info

# Resource usage
sacct -j 12345 --format=JobID,JobName,MaxRSS,Elapsed,State
seff 12345                         # Efficiency summary
```

**Resource Usage Analysis:**

```python
# Python script to analyze SLURM job efficiency
import subprocess
import pandas as pd

def analyze_job_efficiency(job_id):
    """Analyze SLURM job resource efficiency"""
    cmd = f"sacct -j {job_id} --format=JobID,MaxRSS,ReqMem,Elapsed,CPUTime,State --parsable2"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse and analyze efficiency
    lines = result.stdout.strip().split('\n')
    headers = lines[0].split('|')
    data = [line.split('|') for line in lines[1:]]
    
    df = pd.DataFrame(data, columns=headers)
    
    # Calculate memory efficiency
    max_memory = df['MaxRSS'].iloc[0]
    req_memory = df['ReqMem'].iloc[0]
    
    return {
        'job_id': job_id,
        'memory_used': max_memory,
        'memory_requested': req_memory,
        'memory_efficiency': float(max_memory.replace('M','')) / float(req_memory.replace('G','')) * 1000,
        'status': df['State'].iloc[0]
    }
```

### 1.6. Best Practices for Health Data Computing

#### 1.6.1. Pre-Deployment Optimization

- **Profile Before Parallelizing:**
    - Use `cProfile` or `line_profiler` to identify bottlenecks
    - Test on small datasets locally before scaling
    - Measure baseline performance metrics
- **Memory Profiling:**
    - Use `memory_profiler` to track RAM usage
    - Identify memory leaks in long-running processes
    - Optimize data structures for memory efficiency

#### 1.6.2. Data Transfer Optimization

- **Minimize Network I/O:**
    - Co-locate computation with data storage
    - Use compressed formats (Parquet, HDF5) for large datasets
    - Batch small files to reduce metadata overhead
- **Smart Caching:**
    - Cache frequently accessed reference data (genomes, medical ontologies)
    - Use local SSD storage for temporary intermediate results
    - Implement checkpointing for long-running analyses

#### 1.6.3. Error Handling & Monitoring

- **Robust Error Handling:**
    - Implement retry logic for network failures
    - Save intermediate results to prevent data loss
    - Use logging to track job progress and failures
- **Resource Monitoring:**
    - Monitor CPU, memory, and I/O usage during jobs
    - Set up alerts for jobs that exceed expected resource usage
    - Track job efficiency metrics for future optimization

#### 1.6.4. Reproducibility & Documentation

- **Environment Management:**
    - Use containers (Docker/Singularity) for reproducible environments
    - Pin software versions in requirements files
    - Document hardware specifications and performance benchmarks
- **Version Control:**
    - Track analysis scripts and job submission scripts in Git
    - Use semantic versioning for analysis pipelines
    - Document parameter choices and model hyperparameters
- **Resource Documentation:**
    - Record actual vs. requested resources for future jobs
    - Document optimal resource configurations for different analysis types
    - Share resource usage patterns with team members

#### 1.6.5. Health Data Security & Compliance

- **Data Protection:**
    - Encrypt sensitive health data at rest and in transit
    - Use secure file transfer protocols (SFTP, HTTPS)
    - Implement access controls and audit logging
- **HIPAA Compliance:**
    - Ensure PHI is properly de-identified before analysis
    - Use approved computing environments for sensitive data
    - Document data handling procedures for compliance audits
- **Data Retention:**
    - Follow institutional policies for data retention
    - Implement secure data deletion procedures
    - Archive important results following best practices

## 🛠️ Demo Break 1: Distributed Computing in Practice

**Objective:** Understand and run a parallel computation locally, then map to UCSF Wynton/SLURM.

**Steps:**

- Run a parallel patient analysis using `multiprocessing.Pool`
- Modify number of processes and observe speedup
- Discuss how to submit a similar job on Wynton (SGE/SLURM)
- Identify a health data task that would benefit from distributed computing

**Success Criteria:** Output shows parallel speedup; students can describe how to adapt code for Wynton/SLURM.

## 2. Experimental Design & Analysis in Health Data Science

### 2.1. Foundations of Experimental Design

#### 2.1.1. Causal Inference in Health Research

- **Correlation vs. Causation:**
    - **Example:** Coffee consumption and heart disease - confounded by lifestyle factors
    - **Bradford Hill Criteria:** Strength, dose-response, temporality, biological plausibility
    - **Counterfactual Framework:** What would have happened without the intervention?
- **Confounding Variables:**
    - **Definition:** Variables that affect both treatment assignment and outcome
    - **Health Examples:** Age, socioeconomic status, comorbidities, healthcare access
    - **Control Methods:** Randomization, matching, stratification, regression adjustment

#### 2.1.2. Randomized Controlled Trials (RCTs)

- **Design Principles:**
    - **Randomization:** Eliminates selection bias, balances known/unknown confounders
    - **Control Groups:** Placebo, active control, historical control, no treatment
    - **Blinding:** Single-blind (participant), double-blind (participant + researcher), triple-blind (+ analyst)
- **Randomization Methods:**
    - **Simple Randomization:** Coin flip equivalent, may lead to imbalanced groups
    - **Block Randomization:** Ensures balanced groups within time periods
    - **Stratified Randomization:** Balance within important subgroups (age, sex, severity)
    - **Adaptive Randomization:** Adjust probabilities based on interim results

#### 2.1.3. Observational Study Designs

- **Cohort Studies:**
    - **Prospective:** Follow participants forward in time from exposure to outcome
    - **Retrospective:** Look back at historical exposure and outcome data
    - **Health Example:** Framingham Heart Study (cardiovascular risk factors)
- **Case-Control Studies:**
    - **Design:** Compare cases (with disease) to controls (without disease)
    - **Advantages:** Efficient for rare diseases, faster than cohort studies
    - **Health Example:** Smoking and lung cancer association studies
- **Cross-Sectional Studies:**
    - **Design:** Snapshot of population at single time point
    - **Use Cases:** Disease prevalence, risk factor distribution
    - **Limitations:** Cannot establish temporal relationships

#### 2.1.4. Variance Reduction Techniques

- **CUPED (Controlled-experiment Using Pre-Experiment Data):**
    - **Concept:** Use pre-treatment covariates to reduce outcome variance
    - **Formula:** Y_adjusted = Y_post - θ(X_pre - μ_pre)
    - **Health Application:** Use baseline biomarkers to improve treatment effect detection
- **Blocking and Stratification:**
    - **Blocking:** Group similar units before randomization
    - **Stratification:** Analyze subgroups separately, then combine
    - **Health Example:** Stratify by hospital site, disease severity, or demographics
- **Paired Designs:**
    - **Matched Pairs:** Each treatment unit matched with similar control unit
    - **Crossover Designs:** Each participant receives multiple treatments
    - **Health Example:** Twin studies, before/after treatment comparisons

#### 2.1.5. Ethics & Regulatory Considerations

- **Informed Consent:**
    - **Elements:** Risks, benefits, alternatives, right to withdraw
    - **Special Populations:** Minors, cognitively impaired, emergency situations
    - **Digital Health:** Data collection, algorithmic decision-making consent
- **Institutional Review Board (IRB):**
    - **Role:** Review research protocols for ethical compliance
    - **Risk Categories:** Minimal risk, greater than minimal risk, high risk
    - **Health Data:** De-identification requirements, data sharing agreements
- **Data Privacy & Security:**
    - **HIPAA Compliance:** Protected health information (PHI) safeguards
    - **GDPR Considerations:** Right to erasure, data portability, explicit consent
    - **Technical Safeguards:** Encryption, access controls, audit trails

### 2.2. Statistical Methods & Python Tools

#### 2.2.1. Basic Statistical Tests

- **T-Tests:**
    - **One-sample:** Compare sample mean to known value
    - **Two-sample:** Compare means between groups (unpaired/paired)
    - **Assumptions:** Normality, independence, equal variances (for unpaired)
    - **Health Example:** Compare blood pressure before/after medication

**Reference Card: scipy.stats.ttest_ind**

- **Function:** `scipy.stats.ttest_ind(a, b, equal_var=True)`
- **Purpose:** Perform independent two-sample t-test
- **Key Parameters:**
    - `a, b`: (Required) Array-like sample data for two groups
    - `equal_var`: (Optional, default=True) Assume equal population variances
    - `nan_policy`: (Optional, default='propagate') Handle NaN values
    - `alternative`: (Optional, default='two-sided') Type of test ('less', 'greater', 'two-sided')

```python
import scipy.stats as stats
import numpy as np

# Health example: Compare blood pressure reduction
treatment_group = [10, 15, 12, 8, 14, 11, 9, 13]  # mmHg reduction
control_group = [2, 5, 3, 1, 6, 4, 2, 3]          # mmHg reduction

# Perform t-test
t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Significant difference: {p_value < 0.05}")
```

#### 2.2.2. Advanced Statistical Methods

- **ANOVA (Analysis of Variance):**
    - **One-way:** Compare means across multiple groups
    - **Two-way:** Examine two factors and their interaction
    - **Repeated Measures:** Account for within-subject correlation
    - **Health Example:** Compare treatment effects across multiple drug doses

- **Non-parametric Tests:**
    - **Mann-Whitney U:** Non-parametric alternative to t-test
    - **Kruskal-Wallis:** Non-parametric alternative to ANOVA
    - **Wilcoxon Signed-Rank:** Non-parametric paired test
    - **Use Cases:** Non-normal data, ordinal outcomes, small samples

- **Chi-Square Tests:**
    - **Goodness of Fit:** Test if data follows expected distribution
    - **Independence:** Test association between categorical variables
    - **Health Example:** Test association between smoking status and lung disease

#### 2.2.3. Generalized Linear Models (GLMs)

- **Model Components:**
    - **Linear Predictor:** β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
    - **Link Function:** Connects linear predictor to expected outcome
    - **Family Distribution:** Specifies outcome distribution (normal, binomial, Poisson)

- **Common GLM Types:**
    - **Linear Regression:** Continuous outcomes, identity link, normal distribution
    - **Logistic Regression:** Binary outcomes, logit link, binomial distribution
    - **Poisson Regression:** Count outcomes, log link, Poisson distribution
    - **Cox Proportional Hazards:** Time-to-event outcomes, partial likelihood

**Reference Card: statsmodels.formula.api**

```python
import statsmodels.formula.api as smf
import pandas as pd

# Health example: Logistic regression for disease risk
data = pd.DataFrame({
    'disease': [0, 1, 0, 1, 1, 0, 1, 0],
    'age': [45, 62, 38, 58, 67, 41, 55, 49],
    'smoking': [0, 1, 0, 1, 1, 0, 1, 0],
    'bmi': [22.1, 28.5, 21.8, 30.2, 31.1, 23.4, 27.8, 24.6]
})

# Fit logistic regression
model = smf.logit('disease ~ age + smoking + bmi', data=data).fit()
print(model.summary())

# Predict probabilities
prob_disease = model.predict()
print(f"Predicted disease probabilities: {prob_disease}")
```

#### 2.2.4. Multiple Testing Correction

- **Problem:** Increased Type I error rate when conducting multiple tests
- **Family-Wise Error Rate (FWER):** Probability of making ≥1 Type I error
- **False Discovery Rate (FDR):** Expected proportion of false discoveries

- **Correction Methods:**
    - **Bonferroni:** Divide α by number of tests (conservative)
    - **Holm-Bonferroni:** Step-down procedure, less conservative
    - **Benjamini-Hochberg:** Controls FDR instead of FWER
    - **Permutation Tests:** Empirical null distribution from data resampling

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Health example: Multiple biomarker associations
p_values = [0.001, 0.05, 0.02, 0.08, 0.003, 0.12, 0.009, 0.15]
biomarkers = ['CRP', 'IL-6', 'TNF-α', 'LDL', 'HDL', 'Glucose', 'HbA1c', 'Creatinine']

# Apply Benjamini-Hochberg correction
rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

# Display results
for biomarker, p_orig, p_corr, significant in zip(biomarkers, p_values, p_corrected, rejected):
    print(f"{biomarker}: p={p_orig:.3f}, p_corrected={p_corr:.3f}, significant={significant}")
```

#### 2.2.5. Power Analysis & Sample Size Calculation

- **Statistical Power:** Probability of detecting true effect (1 - Type II error rate)
- **Factors Affecting Power:**
    - **Effect Size:** Larger effects easier to detect
    - **Sample Size:** More data increases power
    - **Significance Level:** Lower α reduces power
    - **Variability:** Less noise increases power

```python
from statsmodels.stats.power import ttest_power, tt_solve_power
import matplotlib.pyplot as plt

# Health example: Power analysis for blood pressure study
effect_sizes = np.arange(0.1, 1.5, 0.1)
powers = [ttest_power(effect_size=es, nobs=50, alpha=0.05) for es in effect_sizes]

# Plot power curve
plt.figure(figsize=(10, 6))
plt.plot(effect_sizes, powers, 'b-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Statistical Power')
plt.title('Power Analysis for Blood Pressure Study (n=50 per group)')
plt.grid(True, alpha=0.3)
plt.legend()

# Calculate required sample size for 80% power
required_n = tt_solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Required sample size per group for 80% power: {required_n:.0f}")
```

### 2.3. Health Data Applications

#### 2.3.1. Clinical Trial Design & Analysis

- **Phase I Trials:**
    - **Objective:** Determine safe dosage, identify side effects
    - **Design:** Dose escalation studies, small sample sizes (20-100 participants)
    - **Analysis:** Safety monitoring, dose-limiting toxicity assessment
- **Phase II Trials:**
    - **Objective:** Assess effectiveness while monitoring safety
    - **Design:** Single-arm or randomized studies (100-300 participants)
    - **Analysis:** Response rates, progression-free survival
- **Phase III Trials:**
    - **Objective:** Compare new treatment to standard care
    - **Design:** Large randomized controlled trials (300-3000 participants)
    - **Analysis:** Primary/secondary endpoints, interim analyses, subgroup analyses

#### 2.3.2. Electronic Health Record (EHR) Studies

- **Observational Study Challenges:**
    - **Selection Bias:** Who gets recorded in the EHR?
    - **Information Bias:** Missing data, measurement errors
    - **Confounding:** Indication bias, unmeasured confounders
- **Statistical Approaches:**
    - **Propensity Score Matching:** Balance treatment groups on observed covariates
    - **Instrumental Variables:** Use natural experiments to estimate causal effects
    - **Difference-in-Differences:** Compare changes over time between groups

#### 2.3.3. Genomics & High-Dimensional Data

- **Multiple Testing Challenge:**
    - **GWAS:** Test millions of SNPs simultaneously
    - **RNA-seq:** Test thousands of genes for differential expression
    - **Proteomics:** Analyze hundreds of proteins across conditions
- **Specialized Methods:**
    - **Permutation Tests:** Generate empirical null distributions
    - **Q-values:** Local false discovery rate estimation
    - **Pathway Analysis:** Test gene sets instead of individual genes

#### 2.3.4. Medical Imaging Studies

- **Study Design Considerations:**
    - **Reader Variability:** Multiple radiologists, inter-rater reliability
    - **Image Quality:** Standardize acquisition protocols across sites
    - **Annotation Standards:** Consistent labeling of pathological findings
- **Statistical Approaches:**
    - **ROC Analysis:** Diagnostic performance evaluation
    - **Multi-reader Multi-case (MRMC):** Account for reader and case variability
    - **Hierarchical Models:** Account for clustering (patients within hospitals)

## 🛠️ Demo Break 2: Simulating and Analyzing an A/B Test

**Objective:** Simulate a health intervention A/B test and analyze results using Python.

**Steps:**

- Create or download a CSV with `patient_id`, `group`, `outcome`
- Use `pandas` and `statsmodels` to fit a linear model (`outcome ~ group`)
- Modify group assignments/outcomes and rerun analysis
- Discuss how to interpret results in a health context
- Brainstorm how experimental design would change for a multi-arm or longitudinal study

**Success Criteria:** Students can run the analysis, interpret regression output, and suggest design improvements.

## 3. End-to-End Health Data Science Project Application

### 3.1. Project Lifecycle (CRISP-DM Adapted for Health)

#### 3.1.1. Phase 1: Problem Definition & Scoping

**Stakeholder Identification & Requirements:**

- **Clinical Stakeholders:** Physicians, nurses, researchers, hospital administrators
- **Technical Stakeholders:** Data engineers, ML engineers, IT security, compliance officers
- **Patient Representatives:** Patient advocacy groups, ethics committees
- **Regulatory Bodies:** FDA, IRB, HIPAA compliance officers

**Business Understanding:**

- **Clinical Problem Definition:** What medical question are we addressing?
- **Success Metrics:** Clinical endpoints (mortality, readmission, QoL), operational metrics (cost reduction, efficiency)
- **Constraints:** Budget, timeline, regulatory requirements, data availability
- **Risk Assessment:** Patient safety, data privacy, algorithm bias, clinical workflow disruption

**Experimental Design Integration:**

- **Study Type:** RCT, observational cohort, case-control, cross-sectional
- **Power Analysis:** Sample size calculation based on expected effect size
- **Randomization Strategy:** Simple, block, stratified, adaptive randomization
- **Primary/Secondary Endpoints:** Clinical outcomes, safety measures, quality of life

#### 3.1.2. Phase 2: Data Acquisition & Understanding

**Health Data Sources:**

- **Electronic Health Records (EHR):** Epic, Cerner, Allscripts data extracts
- **Medical Imaging:** DICOM from PACS, pathology slides, radiology reports
- **Laboratory Data:** Blood tests, genetic sequencing, biomarker panels
- **Wearable/IoT Devices:** Continuous glucose monitors, heart rate monitors, activity trackers
- **Public Health Databases:** CDC, WHO, state health departments
- **Research Datasets:** PhysioNet, MIMIC-IV, UK Biobank, All of Us

**Data Quality Assessment:**

- **Completeness:** Missing data patterns, dropout rates, loss to follow-up
- **Accuracy:** Data validation rules, outlier detection, cross-source verification
- **Consistency:** Standardized terminologies (ICD-10, SNOMED, LOINC), unit conversions
- **Timeliness:** Data freshness, lag time between event and recording
- **Privacy Compliance:** De-identification verification, PHI detection, consent validation

**Technical Infrastructure:**

- **Data Storage:** HIPAA-compliant cloud (AWS HIPAA, Azure Healthcare), on-premise servers
- **Access Controls:** Role-based permissions, audit logging, VPN requirements
- **Data Formats:** HL7 FHIR, DICOM, CSV, JSON, Parquet for efficient storage
- **ETL Pipelines:** Apache Airflow, Prefect for automated data processing

#### 3.1.3. Phase 3: Exploratory Data Analysis (EDA)

**Univariate Analysis:**

- **Continuous Variables:** Distributions, outliers, normality testing
- **Categorical Variables:** Frequency tables, missing data patterns
- **Temporal Patterns:** Seasonality, trends, time-to-event distributions
- **Health-Specific Metrics:** Reference ranges, clinical significance thresholds

**Bivariate/Multivariate Analysis:**

- **Correlation Analysis:** Feature relationships, collinearity detection
- **Group Comparisons:** Treatment vs. control, diseased vs. healthy
- **Survival Analysis:** Kaplan-Meier curves, log-rank tests
- **Dimensionality Reduction:** PCA for genomics, t-SNE for clustering

**Hypothesis Generation:**

- **Clinical Questions:** Which biomarkers predict treatment response?
- **Data-Driven Insights:** Unexpected patterns, subgroup identification
- **Confounding Assessment:** Identify potential confounders early
- **Effect Size Estimation:** Preliminary estimates for sample size validation

#### 3.1.4. Phase 4: Data Preparation & Feature Engineering

**Missing Data Handling:**

- **Mechanisms:** Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR)
- **Imputation Strategies:** Mean/median, KNN imputation, multiple imputation, domain-specific defaults
- **Health-Specific Considerations:** Lab values below detection limits, censored survival times

**Feature Engineering:**

- **Temporal Features:** Time since diagnosis, treatment duration, age at onset
- **Clinical Risk Scores:** APACHE II, SOFA score, Framingham Risk Score
- **Derived Biomarkers:** Ratios (LDL/HDL), differences (systolic - diastolic), rates of change
- **Text Features:** NLP on clinical notes, sentiment analysis, named entity recognition

**Data Transformation:**

- **Normalization:** Min-max scaling, z-score standardization, robust scaling
- **Encoding:** One-hot encoding for categories, target encoding for high cardinality
- **Outlier Treatment:** Clinical vs. statistical outliers, domain expertise integration
- **Distribution Transformation:** Log transformation for skewed biomarkers, Box-Cox transforms

#### 3.1.5. Phase 5: Modeling & Intervention Design

**Model Selection Strategy:**

- **Baseline Models:** Logistic regression, linear regression for interpretability
- **Tree-Based Models:** Random forests for tabular health data, XGBoost for competitions
- **Deep Learning:** CNNs for medical imaging, RNNs for time series, transformers for clinical text
- **Ensemble Methods:** Voting classifiers, stacking, blending multiple model types

**Health-Specific Considerations:**

- **Interpretability Requirements:** SHAP values, LIME, feature importance for clinical trust
- **Class Imbalance:** SMOTE, cost-sensitive learning, focal loss for rare diseases
- **Regulatory Compliance:** FDA guidance for AI/ML devices, bias testing, fairness metrics
- **Clinical Integration:** API development, real-time scoring, alert systems

**Experimental Setup:**

- **Train/Validation/Test Splits:** Temporal splits for time series, patient-level splits for longitudinal data
- **Cross-Validation:** Stratified CV for imbalanced classes, group CV for clustered data
- **Hyperparameter Optimization:** Grid search, random search, Bayesian optimization
- **Early Stopping:** Prevent overfitting, monitor validation metrics

#### 3.1.6. Phase 6: Evaluation & Iteration

**Performance Metrics:**

- **Classification:** ROC-AUC, precision-recall curves, sensitivity/specificity for clinical thresholds
- **Regression:** RMSE, MAE, R-squared, clinical significance of errors
- **Survival Analysis:** C-index, Brier score, time-dependent ROC
- **Fairness Metrics:** Equalized odds, demographic parity, individual fairness

**Clinical Validation:**

- **External Validation:** Test on different hospital systems, patient populations
- **Temporal Validation:** Performance over time, concept drift detection
- **Subgroup Analysis:** Performance by age, sex, race, comorbidities
- **Clinical Expert Review:** Sanity checks, domain knowledge validation

**Error Analysis:**

- **Failure Mode Analysis:** When and why does the model fail?
- **Calibration Assessment:** Are predicted probabilities well-calibrated?
- **Feature Attribution:** Which features drive predictions?
- **Bias Detection:** Systematic errors across demographic groups

#### 3.1.7. Phase 7: Communication & Reporting

**Audience-Specific Reporting:**

- **Clinical Audiences:** Focus on clinical significance, patient outcomes, workflow integration
- **Technical Audiences:** Model architecture, performance metrics, computational requirements
- **Regulatory Audiences:** Validation studies, bias testing, safety considerations
- **Executive Audiences:** ROI, operational impact, strategic implications

**Visualization Strategy:**

- **Clinical Dashboards:** Real-time patient monitoring, risk stratification views
- **Research Reports:** Publication-quality figures, statistical test results
- **Interactive Tools:** Altair charts for exploration, Dash apps for stakeholder engagement
- **Automated Reporting:** MkDocs for documentation, GitHub Pages for sharing

#### 3.1.8. Phase 8: Version Control & Reproducibility

**Code Management:**

- **Git Workflows:** Feature branches, pull requests, code review processes
- **Environment Management:** Docker containers, conda environments, requirements.txt
- **Data Versioning:** DVC, MLflow for data and model versioning
- **Experiment Tracking:** Weights & Biases, MLflow, Neptune for experiment management

**Documentation Standards:**

- **Code Documentation:** Docstrings, type hints, README files
- **Analysis Documentation:** Jupyter notebooks with narrative, methodology descriptions
- **Model Documentation:** Model cards, bias testing reports, performance summaries
- **Protocol Documentation:** IRB protocols, statistical analysis plans, data sharing agreements

#### 3.1.9. Phase 9: Deployment & Monitoring

**Deployment Strategies:**

- **Batch Scoring:** Scheduled model runs for risk stratification, population health
- **Real-Time Inference:** API endpoints for clinical decision support, real-time alerts
- **Edge Deployment:** On-device models for wearables, point-of-care testing
- **Federated Learning:** Models trained across institutions without data sharing

**Production Monitoring:**

- **Performance Monitoring:** Model accuracy over time, prediction distribution shifts
- **Data Drift Detection:** Changes in feature distributions, new data patterns
- **Fairness Monitoring:** Ongoing bias detection, equitable performance tracking
- **Clinical Outcome Tracking:** Patient outcomes, intervention effectiveness, safety signals

### 3.2. Health Data Project Examples by Type

#### 3.2.1. Clinical Trial A/B Test Pipeline

**Project Structure:**

- **Power Analysis Phase:** Sample size calculation, effect size estimation, interim analysis planning
- **Randomization System:** Web-based randomization with stratification factors
- **Data Collection:** REDCap integration, mobile app data capture, EHR integration
- **Statistical Analysis:** Intention-to-treat analysis, per-protocol analysis, subgroup analyses
- **Reporting:** CONSORT diagram, primary endpoint analysis, safety reporting

#### 3.2.2. EHR Cohort Study Workflow

**Data Extraction:**

- **Patient Identification:** ICD codes, medication lists, procedure codes
- **Temporal Alignment:** Index dates, follow-up periods, censoring rules
- **Covariate Collection:** Demographics, comorbidities, baseline characteristics
- **Outcome Ascertainment:** Primary endpoints, time-to-event outcomes, composite endpoints

**Analysis Pipeline:**

- **Propensity Score Development:** Logistic regression for treatment assignment
- **Matching/Stratification:** 1:1 matching, inverse probability weighting
- **Survival Analysis:** Cox proportional hazards, competing risks models
- **Sensitivity Analyses:** Unmeasured confounding, missing data assumptions

#### 3.2.3. Genomics Processing Pipeline

**Computational Workflow:**

- **Quality Control:** FastQC for read quality, sample contamination checks
- **Alignment:** BWA, STAR for RNA-seq, reference genome alignment
- **Variant Calling:** GATK HaplotypeCaller, joint genotyping across samples
- **Annotation:** Variant Effect Predictor, population frequency databases
- **Distributed Processing:** SLURM job arrays, containerized workflows

**Statistical Analysis:**

- **Population Structure:** Principal component analysis, admixture analysis
- **Association Testing:** GWAS, gene-based tests, pathway analysis
- **Multiple Testing:** Bonferroni correction, FDR control, permutation testing
- **Functional Follow-up:** eQTL analysis, regulatory annotation, literature review

#### 3.2.4. Medical Imaging ML Workflow

**Image Processing Pipeline:**

- **DICOM Handling:** Anonymization, format conversion, metadata extraction
- **Preprocessing:** Normalization, windowing, artifact removal
- **Augmentation:** Rotation, scaling, elastic deformations for training data
- **Quality Assurance:** Automated quality checks, outlier detection

**Model Development:**

- **Architecture Selection:** ResNet, DenseNet, Vision Transformers
- **Transfer Learning:** ImageNet pretraining, domain adaptation
- **Distributed Training:** Multi-GPU training, gradient accumulation
- **Evaluation:** ROC analysis, CAM visualization, radiologist agreement studies

### 3.2. Detailed Breakdown of Project Stages

1. **Problem Definition & Scoping:**  
   Define questions, stakeholders, success metrics; integrate experimental design principles.  
2. **Data Acquisition & Understanding:**  
   Source health datasets (e.g., PhysioNet/MIMIC-IV); explore data; document variables.  
3. **Exploratory Data Analysis (EDA):**  
   Univariate/bivariate analysis; hypothesis generation; visual diagnostics.  
4. **Data Preparation & Feature Engineering:**  
   Handle missing data; encode categorical features; scale; create time-based and domain-specific features.  
5. **Modeling & Intervention Design:**  
   Select predictive models or design interventions; set up train/validation splits and control groups.  
6. **Evaluation & Iteration:**  
   Apply performance metrics; error analysis; hyperparameter tuning; embed experimental methods (A/B tests, GLMs).  
7. **Communication & Reporting:**  
   Use dashboards, interactive charts, and narrative reports to present findings.  
8. **Version Control & Reproducibility:**  
   Track code and data with Git; document environment and dependencies.  
9. **Deployment & Monitoring:**  
   Brief overview of production deployment; ongoing validation and experiment follow-up.
