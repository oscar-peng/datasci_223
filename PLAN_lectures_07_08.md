# Plan: Lectures 07–08 (LLMs)

> Goal: bring lectures 7 and 8 up to the standard of lectures 4–6. Focus on lecture content only (demos and assignments deferred).

## Content Guidelines

- Follow `CLAUDE.md` (source of truth) and `refs/instructions.md` (supplemental guidance) for all authoring decisions.
- Each major section's **freeform intro** is a great place to repurpose and adapt content from previous years' lectures (`lectures_24/`, `lectures_25/`). Draw analogies, explain concepts, connect to prior lectures — this is where the teaching happens.
- No fluff: no academic discounts sections, no conceptual check-ins, no "what we learned" summaries. Stay practical, pedagogical, and fun.

## Recurring Themes (carry through from lectures 04–06)

- **Bias machines**: neural networks (and LLMs) learn whatever biases exist in their training data and labels. If we're lucky we might guess at the biases we introduce, but not always. This theme runs through the entire NN→LLM series.
- **Know the domain**: if you don't know how to do something yourself, you won't know if an LLM is doing it well. LLMs amplify expertise — they don't replace it.

---

## Context & Constraints

- **Audience**: health data science masters students, Python beginners. They've completed NLP (04), Classification (05), Neural Networks (06).
- **Format**: long-form Markdown, 90-minute sessions, 3 demo breaks each.
- **PyTorch**: defer to computer vision (lecture 9–10). Lectures 7–8 stay framework-light; conceptual code + API usage.
- **Style**: match lectures 4–6 — concept intro → visual/table → reference card → code snippet per major section. No summary sections. XKCD/humor between sections.

---

## High-Level Content Split

### Lecture 07: Transformers & LLM Fundamentals

**Arc**: "You built neural nets in Lecture 6. What happens when you scale them up and add attention?"

1. Bridge from neural nets → transformers
2. Transformer architecture (attention, multi-head, positional encoding)
3. Building a tiny GPT (conceptual + Karpathy's microGPT)
    - **DEMO 1**: microGPT visualizer / attention exploration
4. Embeddings (theory + practical usage, consolidated from 07+08)
5. Prompt engineering fundamentals
    - **DEMO 2**: embedding similarity search + prompt engineering
6. LLM API integration (hands-on setup so students can use APIs in Lecture 8)
    - **DEMO 3**: API prompt engineering with clinical text

### Lecture 08: LLM Applications & Workflows

**Arc**: "Now that you can talk to an LLM, what can you build with it?"

1. Agentic LLMs & tool use
2. RAG (retrieval-augmented generation) — keep focused, not heavy
3. MCP (model context protocol)
    - **DEMO 1**: RAG + MCP hands-on
4. Workflow orchestration patterns (OpenAI Agents SDK examples + Agent Builder GUI)
    - **DEMO 2**: workflow building with Agent Builder
5. When to use LLMs / failure modes / practical recommendations (closing group)
    - **DEMO 3**: practical examples and easy failures

---

## Lecture 07: Detailed Section Outline

### Title line (plain text, no `#`)
Transformers: More than Meets the Eye

### Intro bullets
- Brief topic list (same pattern as lectures 4–6)

### # From Neural Networks to Transformers

Freeform intro adapting content from:
- `lectures_24/07/lecture_07.md` lines 96–134 (history section)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 9–78 (timeline)
- `lectures/06/lecture_06.md` LSTM/RNN content (back-reference)

Content:
- Bridge: "In Lecture 6 you trained dense, CNN, and LSTM networks. LSTMs process sequences one token at a time — what if we could process them all at once?"
- RNN limitations recap (vanishing gradients, sequential bottleneck) — brief, since covered in 06
- The attention breakthrough (2015) → Transformers (2017)
- Timeline table: Word2Vec → Seq2Seq → Attention → Transformer → GPT → ChatGPT → GPT-4/Claude
- **Reference Card: NLP Model Evolution** (timeline table with year, innovation, key insight)

### # Transformer Architecture

Freeform intro adapting from:
- `lectures_24/07/lecture_07.md` lines 100–134 (architecture + attention)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 117–160 (attention, multi-head, components)

Content:
- High-level diagram (encoder-decoder, or decoder-only for GPT)
- **Self-attention**: how tokens attend to each other
- **Multi-head attention**: why multiple "perspectives" matter
- Key components: embeddings, positional encoding, feed-forward, layer norm, residual connections
- Scaled dot-product attention formula (Q, K, V)
- Student-facing links to visual explainers:
    - Transformer Explainer (interactive)
    - Jay Alammar's "The Illustrated Transformer" (visual walkthrough)
    - krupadave.com "Everything About Transformers" (story-driven)
- **Reference Card: Transformer Components** (multi-method table)
- **Reference Card: Scaled Dot-Product Attention** (single deep-dive)
- **Code Snippet: Simplified Attention** (pure Python/numpy, no framework)

### # Building a GPT from Scratch

Freeform intro adapting from:
- `lectures_24/07/lecture_07.md` lines 183–261 (fine-tuning walkthrough — reframe as "what's inside")
- Karpathy microGPT blog post (new resource)

Content:
- Conceptual walkthrough: what does it take to build a language model?
- Karpathy's microGPT (200-line, zero-dependency GPT) — explain the key pieces:
    - Tokenization (character-level)
    - Autograd engine
    - Multi-head attention blocks
    - Training loop (forward, loss, backprop, Adam)
    - Inference/sampling with temperature
- How this scales to production (what changes: tokenizer, data, compute — "the core algorithm doesn't")
- **Theme tie-in**: the model learns from its training data. All of it. Including the biases.
- **Reference Card: GPT Components** (single deep-dive: tokenizer, embedding, attention, FFN, output)
- **Code Snippet: Minimal Attention Block** (annotated, from microGPT)

### # LIVE DEMO!
- Exploring attention visualization (microGPT visualizer / nanoGPT)

### # Embeddings

Freeform intro adapting from:
- `lectures/07/lecture_07.md` lines 146–223 (current embeddings section)
- `lectures/08/lecture_08.md` lines 134–237 (embeddings in practice — consolidate here)
- `lectures_24/07/lecture_07.md` lines 154–167 (embeddings)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 106–115 (use cases)
- `lectures/04/lecture_04.md` word vectors section (back-reference, not duplication)

Content:
- What embeddings are: mapping discrete tokens to continuous vectors
- Word2Vec, GloVe, FastText (historical context, connect back to Lecture 04)
- Modern embeddings: BERT, Sentence Transformers
- Latent space: the geometric world where meaning lives (king − man + woman ≈ queen)
- Practical usage (consolidated from lecture 08):
    - Semantic search
    - Document clustering
    - Similarity matching
    - Anomaly detection
- Sentence Transformers for computing embeddings
- Cosine similarity for comparing them
- Vector databases overview (ChromaDB, FAISS, Pinecone) — brief intro, RAG in lecture 08 will build on this
- **Reference Card: Common Embedding Methods** (multi-method table)
- **Reference Card: `SentenceTransformer`** (single deep-dive)
- **Reference Card: `cosine_similarity`** (single deep-dive)
- **Code Snippet: Computing and Comparing Embeddings**
- **Code Snippet: ChromaDB Vector Search** (moved from lecture 08)

### # LLMs and General-Purpose Models

Freeform intro adapting from:
- `lectures_24/07/lecture_07.md` lines 169–191 (LLMs, hallucination)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 241–255 (fine-tuning vs prompt engineering)

Content:
- What makes an LLM "general purpose" (pre-training on massive corpora, emergent capabilities)
- Fine-tuning vs prompt engineering: when to use each (comparison table)
- Addressing hallucination (brief — expanded in lecture 08 failure modes)
- **Theme tie-in**: "if you don't know how to do something, you won't know if an LLM is doing it well"

### # Prompt Engineering

Freeform intro adapting from:
- `lectures/07/lecture_07.md` lines 298–386 (current prompt engineering section)
- `lectures_24/07/lecture_07.md` lines 262–298 (structured responses, one/few-shot)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 256–303 (few-shot, prompt example)

Content:
- Zero-shot, one-shot, few-shot learning
- Structured responses: why they matter for health data (JSON schemas, function calling)
- Schema-based prompting
- **Reference Card: Prompting Techniques** (multi-method table)
- **Reference Card: Structured Output Prompting** (single deep-dive)
- **Code Snippet: Few-Shot Prompting**
- **Code Snippet: Schema-Based Prompting**

### # LIVE DEMO!!
- Embedding similarity search + prompt engineering techniques

### # LLM API Integration

Freeform intro adapting from:
- `lectures/07/lecture_07.md` lines 388–551 (current API section)
- `lectures_24/07/lecture_07.md` lines 300–562 (API providers, function calling, LLM client, CLI)

Content:
- REST API patterns (API keys, rate limiting, error handling)
- Provider comparison table (OpenAI, Anthropic, Google, Hugging Face)
- Function calling for schema compliance
- Building a reusable LLM client (class pattern with retry, conversation history)
- **Reference Card: LLM API Providers** (multi-method table)
- **Reference Card: Function Calling** (single deep-dive)
- **Code Snippet: OpenAI API**
- **Code Snippet: Function Calling**
- **Code Snippet: LLM Client Class**

### # LIVE DEMO!!!
- Zero-, one-, and few-shot prompting via API (clinical text examples)

### # Resources and Links
- Transformers & attention
- Building GPTs
- LLMs & healthcare AI
- Prompt engineering guides
- Where to play around

---

## Lecture 08: Detailed Section Outline

### Title line (plain text, no `#`)
LLM Applications & Workflows

### Intro bullets
- Brief topic list

### # Agentic LLMs

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 417–516 (current agents section)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 305–345 (agentic LLMs, prompting techniques)

Content:
- Traditional LLM use vs agentic: comparison table
- Key characteristics (autonomy, tool use, iteration, state management)
- Agent components (planner, memory, tools, executor, reflector)
- Agent loop: plan → act → observe → reflect
- Prompting techniques for agents:
    - Chain-of-thought
    - Self-consistency
    - ReAct (reason + act)
    - Reflection
- Important caveat: "reasoning" ≠ thinking; doesn't always improve results; always more expensive
    - Link: Apple "Illusion of Thinking" research
- **Theme tie-in**: agents inherit all the biases of the underlying model, plus whatever biases the tool selection and prompt design introduce
- **Reference Card: Agent Components** (multi-method table)
- **Reference Card: Advanced Prompting Patterns** (multi-method table)
- **Code Snippet: Simple Agent Loop**

### # Retrieval-Augmented Generation (RAG)

Keep this focused — explain the pattern, show the pipeline, one code example. Don't go deep on chunking strategies or best practices.

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 240–319 (current RAG section — trim)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 264–283 (RAG overview)

Content:
- Why RAG? (reduces hallucinations, provides sources, keeps info current, domain adaptation)
- RAG pipeline diagram: Query → Embed → Retrieve → Augment prompt → Generate
- Connection to embeddings + vector DBs from lecture 07
- **Reference Card: RAG Pipeline** (single deep-dive)
- **Code Snippet: Simple RAG Pipeline**

### # Model Context Protocol (MCP)

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 329–415 (current MCP section)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 360–371 (MCP)

Content:
- What MCP standardizes: connecting LLMs to external tools/data sources
- MCP concepts: server, tool, resource, transport
- Common MCP servers table (filesystem, postgres, github, puppeteer)
- How MCP fits with agents: tools that agents can call
- **Reference Card: MCP Concepts** (multi-method table)
- **Code Snippet: Using MCP with OpenAI**

### # LIVE DEMO!
- RAG pipeline + MCP integration hands-on

### # Workflow Orchestration Patterns

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 518–703 (current workflows section)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 373–479 (workflow patterns)

Content:
- Why workflows matter (reliability, cost control, auditability)
- **Pattern: Prompt Chaining** — sequential LLM calls, each building on the last
- **Pattern: Guardrails** — PHI/PII detection, hallucination checks, format validation
- **Pattern: Deterministic Steps** — combining LLMs with rule-based logic (e.g., dose calculations)
- Advanced patterns (brief mentions):
    - Orchestrator-workers
    - Evaluator-optimizer
    - Routing & logic
    - Human-in-the-loop
    - Parallelization (fan-out/fan-in)

#### Agent/Workflow Frameworks

| Framework | Focus | Notes |
|-----------|-------|-------|
| **OpenAI Agents SDK** | Agent building with tools, handoffs, guardrails, tracing | Primary framework for this course. Has Agent Builder GUI. |
| **LangChain / LangGraph** | Chains, agents, stateful graphs | Widely used, steeper learning curve. Good for custom workflows. |
| **AutoGen** (Microsoft) | Multi-agent conversations | Research-oriented, good for multi-agent patterns |
| **smolagents** (Hugging Face) | Lightweight agents | Minimal, good for quick prototyping |
| **Anthropic Claude Code / claude-flow** | CLI-based agentic coding | Developer tooling focus |
| **AI SDK** (Vercel) | Web-integrated agents | TypeScript-first, good for web apps |

- **Primary focus: OpenAI Agents SDK**
    - Core concepts: Agent, Runner, Tools, Handoffs, Guardrails, Tracing
    - Agent Builder GUI for visual workflow design
- **Runner-up: LangChain/LangGraph**
    - Brief mention, link to example project (`abe_froman` — human-readable custom workflows)
- **Reference Card: Workflow Patterns** (multi-method table)
- **Reference Card: Common Guardrails** (multi-method table)
- **Code Snippet: Prompt Chain** (OpenAI)
- **Code Snippet: Guardrails (PHI detection)**
- **Code Snippet: OpenAI Agents SDK basic agent**

### # LIVE DEMO!!
- Workflow building with OpenAI Agent Builder GUI + Agents SDK

### # When to Use LLMs

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 11–51 (current when-to-use section)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 197–216 (when to use)

Content:
- Good fits (summarization, extraction, classification, Q&A, draft generation)
- Poor fits (calculations, factual retrieval, real-time data, autonomous clinical decisions, deterministic logic)
- Decision framework
- **Theme tie-in**: "if you don't know how to do something, you won't know if an LLM is doing it well"

### # Common Failure Modes

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 53–126 (current failure modes)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 217–237 (failure modes)

Content:
- Hallucinations → RAG, fact-checking, citations, temperature=0
- Prompt injection → input sanitization, delimiters, XML tags
- Inconsistency → seeded states, validation
- Context overflow → strategic positioning, chunking, hierarchical summarization
- Task/expertise mismatch → expert review, reference materials
- **Reference Card: Failure Modes & Mitigations** (multi-method table)

### # Practical Recommendations

Freeform intro adapting from:
- `lectures/08/lecture_08.md` lines 712–745 (current practical recs)
- `lectures_25/2025/🌪️ Whirlwind Tour...` lines 484–523 (getting started, testing)

Content:
- Start small: baby models for prototyping, cost comparison table
- Self-hosted options (Ollama, PocketPal) — brief, practical
- Testing & validation: how to verify LLM outputs
- Red flags to watch for
- **Theme tie-in (closing)**: these are bias machines. They learn from whatever data/labels we give them. If we're lucky, we might guess at the biases we introduce — but not always. Domain expertise is the irreplaceable ingredient.

### # LIVE DEMO!!!
- Practical examples and easy failures (hallucination demos, prompt injection, showing where LLMs break)

### # Resources
- Prompt engineering guides
- Workflow tools & agent frameworks
- Self-hosting options
- Healthcare AI tools

---

## Key Changes from Current State

### Lecture 07 changes

| Change | Rationale |
|--------|-----------|
| **Add "From Neural Networks to Transformers" bridge section** | Lectures 4–6 build on each other; 07 should explicitly connect to 06 |
| **Add "Building a GPT from Scratch" section** | Karpathy's microGPT is a perfect conceptual anchor; demystifies what's inside an LLM |
| **Consolidate embeddings** (pull practical usage from 08 into 07) | Keeps embedding theory + practice together; frees space in 08 |
| **Swap Demo 2 (model selection) out** | Model selection with cross-validation is a lecture 05 topic |
| **Demo placement**: after GPT, after embeddings+prompting, after API | Front-loads conceptual content, demos follow naturally |
| **Keep API integration in 07** | Students need API access set up before lecture 08's applied work |
| **Add external resource links** | microGPT blog, microGPT visualizer, krupadave, Alammar illustrated transformer |

### Lecture 08 changes

| Change | Rationale |
|--------|-----------|
| **Lead with Agentic LLMs** | Sets the frame: "LLMs that take action" — then RAG/MCP/workflows are tools agents use |
| **Trim RAG** | Was too heavy; keep it focused on the pattern + one code example |
| **Move embeddings-in-practice to lecture 07** | Consolidation; 08 just references "embeddings from last lecture" in RAG section |
| **Add frameworks comparison table** | Students should know the landscape; focus on OpenAI Agents SDK |
| **Replace raw asyncio with Agents SDK** | More practical, matches industry tooling, has GUI for demo |
| **Move "When to Use" / failure modes / practical recs to end** | Better as closing wisdom after students have seen what's possible |
| **Add recurring themes** | Bias machines, domain expertise, "illusion of thinking" |
| **Remove fluff** | No academic discounts, no conceptual check-ins |

### Demo realignment

| Lecture | Demo 1 | Demo 2 | Demo 3 |
|---------|--------|--------|--------|
| **07** | microGPT / attention viz | Embedding search + prompting | API prompt engineering (clinical text) |
| **08** | RAG + MCP hands-on | Workflow building (Agent Builder) | Practical failures (hallucination, injection) |

---

## Source Material Reference

### Internal repo sources

| Source | Path | Content to adapt |
|--------|------|------------------|
| Current lecture 07 | `lectures/07/lecture_07.md` | Transformer architecture, attention code, embeddings, prompt engineering, API integration, resources |
| Current lecture 08 | `lectures/08/lecture_08.md` | Agentic LLMs, RAG, MCP, workflow patterns, failure modes, practical recs |
| 2024 lecture 07 | `lectures_24/07/lecture_07.md` | Fine-tuning walkthrough (lines 183–261), LLM client class (lines 409–492), CLI (lines 494–549), API examples (lines 300–399), transformer history (lines 96–134) |
| 2025 Whirlwind Tour | `lectures_25/2025/🌪️ Whirlwind Tour of LLMs...md` | History timeline (lines 9–78), technical foundations (lines 79–160), when-to-use/failure modes (lines 197–238), workflow patterns (lines 397–479), getting started/testing (lines 484–523) |
| 2024 demo 03 | `lectures_24/07/demo/03-api_prompt_engineering.md` | Healthcare prompt engineering examples (classify_medical_text, generate_medical_report) |
| 2024 demo 02 | `lectures_24/07/demo/02-nanogpt_attention.md` | nanoGPT attention visualization walkthrough |
| Lecture 04 (NLP) | `lectures/04/lecture_04.md` | Word vectors/embeddings intro (back-reference, not duplication) |
| Lecture 06 (NN) | `lectures/06/lecture_06.md` | LSTM/RNN content (bridge section back-reference), bias/tank detector parable |
| `abe_froman` project | https://github.com/christopherseaman/abe_froman | LangGraph example — human-readable custom workflows using Claude Code |

### External resources

| Resource | URL | Use in lectures |
|----------|-----|-----------------|
| **Transformers & Architecture** | | |
| The Illustrated Transformer (Jay Alammar) | https://jalammar.github.io/illustrated-transformer/ | Lecture 07 — progressive visual walkthrough of full architecture |
| "Everything About Transformers" (Krupa Dave) | https://www.krupadave.com/articles/everything-about-transformers | Lecture 07 — visual story-driven reference |
| Transformer Explainer (interactive) | https://poloclub.github.io/transformer-explainer/ | Lecture 07 — interactive tool |
| "Attention is All You Need" paper | https://arxiv.org/abs/1706.03762 | Lecture 07 — original paper reference |
| Attention mechanism paper (2015) | https://arxiv.org/abs/1409.0473 | Lecture 07 — history section |
| Visual intro to Attention | https://erdem.pl/2021/05/introduction-to-attention-mechanism | Lecture 07 — supplementary visual |
| Multi-head attention deep dive | https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853 | Lecture 07 — supplementary |
| Building Transformers from Scratch | https://vectorfold.studio/blog/transformers | Lecture 07 — supplementary |
| **Building GPTs** | | |
| Karpathy microGPT blog | https://karpathy.github.io/2026/02/12/microgpt/ | Lecture 07 — conceptual walkthrough of 200-line GPT |
| microGPT visualizer (Boratto) | https://microgpt.boratto.ca | Lecture 07 demo 1 — interactive visualization of GPT internals |
| nanoGPT repo | https://github.com/karpathy/nanoGPT | Lecture 07 resources |
| Karpathy Zero to Hero | https://karpathy.ai/zero-to-hero.html | Lecture 07 resources |
| Let's Build GPT (YouTube) | https://www.youtube.com/watch?v=kCc8FmEb1nY | Lecture 07 resources |
| GPT-2 WebGL visualizer | https://github.com/nathan-barry/gpt2-webgl | Lecture 07 resources |
| **LLMs & Healthcare AI** | | |
| UCSF Versa | https://ai.ucsf.edu/platforms-tools-and-resources/ucsf-versa | Lecture 07/08 resources — local institutional tool |
| Suki AI | https://www.suki.ai/ | Lecture 08 resources — clinical AI assistant |
| Google Med-PaLM | https://sites.research.google/med-palm/ | Lecture 08 resources |
| Apple "Illusion of Thinking" | https://machinelearning.apple.com/research/illusion-of-thinking | Lecture 08 agents section — LLM reasoning limitations |
| **Prompt Engineering** | | |
| Anthropic prompt engineering guide | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering | Lecture 07 resources |
| OpenAI prompt engineering guide | https://platform.openai.com/docs/guides/prompt-engineering | Lecture 07 resources |
| OpenAI prompt examples | https://platform.openai.com/docs/examples | Lecture 07 resources |
| **Agent & Workflow Frameworks** | | |
| OpenAI Agents SDK (Python) | https://github.com/openai/openai-agents-python | Lecture 08 — primary framework |
| OpenAI Agents SDK docs | https://openai.github.io/openai-agents-python | Lecture 08 — reference |
| OpenAI Agent Builder GUI | https://platform.openai.com/agent-builder | Lecture 08 demo 2 — visual workflow builder |
| OpenAI Agents guide | https://platform.openai.com/docs/guides/agents | Lecture 08 — reference |
| LangChain docs | https://python.langchain.com/docs | Lecture 08 — runner-up framework |
| LangGraph | https://www.langchain.com/langgraph | Lecture 08 — runner-up framework |
| `abe_froman` (LangGraph example) | https://github.com/christopherseaman/abe_froman | Lecture 08 — human-readable custom workflow example |
| AutoGen (Microsoft) | https://microsoft.github.io/autogen/stable//index.html | Lecture 08 — framework table |
| smolagents (Hugging Face) | https://huggingface.co/docs/smolagents/index | Lecture 08 — framework table |
| AI SDK (Vercel) | https://ai-sdk.dev/docs/agents/overview | Lecture 08 — framework table |
| **MCP** | | |
| MCP documentation | https://modelcontextprotocol.io | Lecture 08 MCP section |
| MCP servers repo | https://github.com/modelcontextprotocol/servers | Lecture 08 MCP section |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk | Lecture 08 MCP section |
| **Self-Hosting & Tools** | | |
| Ollama | https://ollama.com | Lecture 08 practical recs |
| PocketPal | https://github.com/a-ghorbani/pocketpal-ai | Lecture 08 practical recs |
| IBM Granite 4.0 | https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models | Lecture 08 resources |
| OpenAI open-source models (gpt-oss) | https://openai.com/index/introducing-gpt-oss/ | Lecture 08 resources |
| **Workflow Orchestrators** | | |
| Kestra | https://kestra.io | Lecture 08 resources |
| Inngest | https://www.inngest.com | Lecture 08 resources |
| Temporal | https://temporal.io | Lecture 08 resources |
| **Developer Tools** | | |
| Claude Code | https://www.claude.com/product/claude-code | Lecture 08 resources |
| Cursor | https://cursor.com/ | Lecture 08 resources |
| OpenAI Codex | https://openai.com/codex/ | Lecture 08 resources |
| **Papers** | | |
| GPT (2018) paper | https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf | Lecture 07 resources |
| RLHF paper | https://arxiv.org/abs/2203.02155 | Lecture 07 resources |
| DistilBERT paper | https://arxiv.org/pdf/1910.01108v4.pdf | Lecture 07 resources (knowledge distillation) |
| **Cookbooks** | | |
| Anthropic Cookbook | https://github.com/anthropics/anthropic-cookbook | Lecture 08 resources |
| OpenAI Cookbook | https://cookbook.openai.com/ | Lecture 08 resources |
| **Platforms** | | |
| OpenAI Platform | https://platform.openai.com/ | Lecture 07 resources |
| Hugging Face NLP Course | https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt | Lecture 07 resources |
| Google Vertex AI | https://cloud.google.com/vertex-ai | Lecture 07 resources |
| OpenAI Evals | https://github.com/openai/evals | Lecture 08 resources |
| Open LLMs list | https://github.com/eugeneyan/open-llms | Lecture 07 resources |

---

## Quality Checklist (to validate against lectures 4–6 standard)

Per `CLAUDE.md` and `refs/instructions.md`:

- [ ] Title is plain text (no `#`)
- [ ] Every major section has: freeform intro → visual/table → Reference Card → Code Snippet
- [ ] Freeform intros draw from previous years' content where applicable
- [ ] Reference cards use standard formats (multi-method table or single-function deep-dive)
- [ ] Code snippets are short, focused, runnable, with health data examples
- [ ] Exactly 3 demo markers: `# LIVE DEMO!`, `# LIVE DEMO!!`, `# LIVE DEMO!!!`
- [ ] Humor/visuals placed between sections (not inside core explanations)
- [ ] No time estimates
- [ ] No summary sections
- [ ] No fluff (academic discounts, conceptual check-ins)
- [ ] Images use relative paths (`media/...`)
- [ ] No `#FIXME` markers remain (or explicitly flagged for follow-up)
- [ ] Resources section at end with categorized links
- [ ] Recurring themes woven in: bias machines, domain expertise required
