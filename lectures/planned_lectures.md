# Lecture Plan (11 Lectures)

## Foundational (L01–L04)

### L01: Setup + Debugging ✅ *in progress*
- Compressed dev env setup (prereq covered basics)
- Notebook hygiene and reproducibility
- Defensive programming, debugging in VS Code + Jupyter
- Adapt from: `lectures_25/01` (setup), `lectures_25/02` (debugging)

### L02: SQL for Data Analysis
- Core SQL: SELECT, JOIN, GROUP BY, window functions
- SQLite + pandas/sqlalchemy integration
- Adapt from: `lectures_25/03` (was L03 last year)

### L03: Larger-than-Memory Data
- Polars: lazy vs eager evaluation
- Out-of-core processing, chunking, parquet
- Database as one solution for large data (ties to L02)
- Adapt from: `lectures_25/02` (was L02 last year—order swapped)

### L04: NLP Foundations *(student request)*
- Text preprocessing, tokenization, embeddings
- Sentiment, NER, text classification
- Clinical/health text applications
- Sets up context for transformer/LLM lectures
- New content + foundations from `lectures_25/07`

---

## ML/AI Progression (L05–L09)

### L05: Classification
- Train/test splits, evaluation metrics, cross-validation
- Logistic regression → Random Forest → XGBoost
- Handling imbalanced data, feature selection
- Adapt from: `lectures_25/05`

### L06: Neural Networks
- MLP, CNN, RNN/LSTM fundamentals (PyTorch)
- Training loop, backprop, regularization
- Adapt from: `lectures_25/06`

### L07: Transformers & Deep Learning
- Attention mechanism, transformer architecture
- Hugging Face basics, tokenization
- Adapt from: `lectures_25/07`

### L08: LLMs – DIY & Understanding
- Building intuition: nanoGPT walkthrough
- Embeddings, context windows, fine-tuning concepts
- Adapt from: `lectures_25/07` demos

### L09: LLMs – API, Agentic & Workflows
- OpenAI/Anthropic/local APIs
- Prompt engineering, structured outputs
- Tool use, agents, practical workflows
- Adapt from: `lectures_25/07` + new content

---

## Applied / Student Choice (L10–L11)

### L10: Time Series & Forecasting
- Time-based splits, lag features, rolling windows
- ARIMA basics, ML regressors for forecasting
- Adapt from: `lectures_25/04`

### L11: TBD – Student Vote
Candidates:
- **Computer Vision** – CNNs, transfer learning, medical imaging (from `lectures_25/08`)
- **Visualization & Dashboards** – Altair, Streamlit, MkDocs reports (from `lectures_25/09`)
- **Experimentation & A/B Testing** – causal inference, power analysis (from `lectures_25/10`)
- **Distributed Computing** – threads/processes, HPC intro
- **End-to-End Project** – CRISP-DM, capstone guidance

---

## Notes
- Each lecture: 90 min + additional demo time at 1/3, 2/3, end
- Demos in Jupyter (draft as markdown, convert via jupytext)
- Assignments: pass/fail, test understanding of core lecture content
- Reference last year's content in `lectures_25/` for examples, datasets, demos
