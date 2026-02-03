LLM Applications & Workflows

- When to Use LLMs
- Common Failure Modes and Mitigations
- Embeddings for Semantic Search
- Retrieval-Augmented Generation (RAG)
- Agentic LLMs
- Workflow Orchestration Patterns
- Model Context Protocol (MCP)

# When to Use LLMs

Building on our understanding of transformers and LLM APIs from last week, we now focus on practical applications. Knowing **when** to use LLMs is just as important as knowing **how**.

![LLM tools landscape](media/mediaagents_landscape.png)

*The LLM application landscape includes agents, RAG systems, workflow orchestrators, and more.*

## Good Fits for LLMs

**Text summarization and transformation**: Condense documents while preserving key information

**Structured data extraction**: Convert unstructured text to structured formats (JSON, tables)

**Content classification**: Categorize by type, topic, sentiment

**Question answering over documents**: Answer questions based on provided context

**Draft generation with review**: First drafts that humans refine

## Poor Fits for LLMs

**Precise calculations**: Use tools (calculators, code) instead

**Factual retrieval without verification**: LLMs may hallucinate

**Real-time data without external connection**: Models have knowledge cutoffs

**High-stakes autonomous decisions**: Require human oversight

**Deterministic logic**: Use rule engines instead

### Decision Framework

Ask yourself:

- Can you describe the task clearly?
- Are errors catchable?
- Can you validate outputs?

If "yes" to all three, LLMs may be a good fit. If "no" to any, consider alternatives or add safeguards.

# Common Failure Modes

Understanding how LLMs fail helps you design better systems and set appropriate expectations.

## Hallucinations

**What**: Fabricated citations, confident incorrect answers, plausible-sounding but false information

**Why**: Models generate statistically likely continuations, not verified facts

**Mitigation strategies**:

- RAG (Retrieval-Augmented Generation) - ground responses in retrieved documents
- Fact-checking pipelines
- Require citations with verification
- Use lower temperature for factual tasks

### Reference Card: Hallucination Mitigation

| Strategy | Description | When to Use |
|:---|:---|:---|
| **RAG** | Ground responses in retrieved documents | Document-based Q&A |
| **Fact-checking** | Verify claims against trusted sources | High-stakes applications |
| **Citation requirement** | Ask model to cite sources | Research assistance |
| **Temperature=0** | Reduce randomness | Extraction tasks |

## Prompt Injection

**What**: User input overrides system instructions, causing unintended behavior

**Why**: Models may treat user content as instructions

**Mitigation strategies**:

- Separate user content from system instructions clearly
- Input sanitization
- Output filtering
- Use delimiters to mark user content (e.g., `"""User input: {input}"""` or XML tags like `<user_input>...</user_input>`)

## Inconsistency

**What**: Same input produces different outputs (when temperature > 0)

**Why**: Sampling introduces randomness

**Mitigation strategies**:

- Set `temperature=0` for extraction tasks
- Use seeded random states where supported
- Implement validation and retry logic

## Context Overflow

**What**: Important information at edges of context gets lost or ignored

**Why**: Attention mechanisms may not weight all positions equally

**Mitigation strategies**:

- Place critical information at start and end
- Chunk long documents strategically
- Use hierarchical summarization for very long inputs

## Task/Expertise Mismatch

**What**: User lacks domain knowledge to identify LLM errors

**Why**: LLMs are confident even when wrong

**Mitigation strategies**:

- Require expert review for domain-specific outputs
- Provide reference materials
- Limit autonomous decisions

# LIVE DEMO!

Embedding similarity search for semantic document retrieval.

See: [demo/01_embedding_search.md](demo/01_embedding_search.md)

# Embeddings in Practice

Building on the embedding concepts from last week, let's see how they enable powerful applications.

## Embedding Use Cases

- **Semantic search**: Find similar documents by meaning, not just keywords
- **Document clustering**: Group related documents automatically
- **Similarity matching**: Find duplicates, related items, or anomalies
- **Anomaly detection**: Identify outliers in embedding space
- **Classification**: Use embeddings as features for downstream models

## Creating and Using Embeddings

### Reference Card: Sentence Transformers

| Component | Details |
|:---|:---|
| **Library** | `sentence-transformers` |
| **Purpose** | Generate dense vector embeddings for sentences and documents |
| **Popular Models** | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |
| **Output** | NumPy array of shape (n_sentences, embedding_dim) |
| **Install** | `pip install sentence-transformers` |

### Code Snippet: Embedding Documents

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your documents
documents = [
    "Patient presents with chest pain and shortness of breath",
    "Lab results show elevated troponin levels",
    "Recommend cardiac catheterization",
    "Patient reports headache and nausea"
]

# Generate embeddings
embeddings = model.encode(documents)

# Find similar documents to a query
query = "cardiac symptoms"
query_embedding = model.encode([query])

# Calculate similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]
most_similar_idx = np.argmax(similarities)
print(f"Most similar: {documents[most_similar_idx]}")
```

## Vector Databases

For production applications with many documents, you'll want a vector database:

### Reference Card: Vector Database Options

| Database | Type | Strengths |
|:---|:---|:---|
| **ChromaDB** | In-memory/persistent | Simple API, good for prototyping |
| **FAISS** | In-memory | Fast, scalable, from Facebook AI |
| **Pinecone** | Cloud service | Managed, production-ready |
| **Weaviate** | Self-hosted/cloud | Full-text + vector search |
| **pgvector** | PostgreSQL extension | Integrate with existing DB |

### Reference Card: ChromaDB API

| Method | Purpose | Key Parameters |
|:---|:---|:---|
| `chromadb.Client()` | Create in-memory client | — |
| `client.create_collection(name)` | Create a new collection | `name` (str), `metadata` (dict) |
| `collection.add()` | Add documents | `documents`, `ids`, `embeddings`, `metadatas` |
| `collection.query()` | Search similar documents | `query_embeddings`, `n_results`, `include` |
| `collection.count()` | Get document count | — |

### Code Snippet: ChromaDB for Vector Search

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("clinical_notes")

# Add documents with embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["Note 1...", "Note 2...", "Note 3..."]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=model.encode(documents).tolist()
)

# Query
results = collection.query(
    query_embeddings=model.encode(["chest pain symptoms"]).tolist(),
    n_results=3
)
```

# Retrieval-Augmented Generation (RAG)

RAG combines the power of retrieval systems with generative models, grounding LLM responses in actual documents.

## Why RAG?

- **Reduces hallucinations**: Responses grounded in retrieved documents
- **Provides sources**: Can cite specific documents
- **Keeps information current**: Update documents without retraining
- **Domain adaptation**: Use your own documents without fine-tuning

## RAG Pipeline

![RAG pipeline diagram](media/rag_pipeline.png)

```
Query → Embed → Retrieve Similar Chunks → Add to Prompt → Generate Response
```

### Reference Card: RAG Components

| Component | Purpose | Tools |
|:---|:---|:---|
| **Document Loader** | Ingest documents | LangChain loaders, PyPDF |
| **Text Splitter** | Chunk documents | Manual slicing, LangChain splitters |
| **Embedding Model** | Vectorize chunks | Sentence Transformers, OpenAI |
| **Vector Store** | Store and retrieve | ChromaDB, FAISS, Pinecone |
| **LLM** | Generate response | OpenAI, Anthropic, local models |

> **Chunking tip**: For simple cases, split on paragraph boundaries or fixed character counts. Libraries like LangChain provide `RecursiveCharacterTextSplitter` for smarter splitting that respects sentence boundaries.

## Building a RAG Pipeline

### Code Snippet: Simple RAG

```python
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# Setup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_client = OpenAI()
db = chromadb.Client()
collection = db.create_collection("docs")

# Index documents
def index_documents(documents):
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

# RAG query
def rag_query(question, n_results=3):
    # Retrieve relevant chunks
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    
    # Build context
    context = "\n\n".join(results['documents'][0])
    
    # Generate response with context
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content
```

## RAG Best Practices

1. **Chunk size matters**: Balance between context and specificity (typically 500-1000 tokens)
2. **Overlap chunks**: Include overlap to avoid splitting important information
3. **Metadata**: Store document source, date, and other metadata for filtering
4. **Hybrid search**: Combine semantic search with keyword search
5. **Reranking**: Use a reranker model to improve retrieval quality

# LIVE DEMO!!

Building a simple RAG pipeline for clinical document Q&A.

See: [demo/02_rag_pipeline.md](demo/02_rag_pipeline.md)

# Model Context Protocol (MCP)

MCP provides a standardized way to connect LLMs to external data sources and tools. Instead of writing custom integrations for each tool, MCP offers plug-and-play servers that expose capabilities in a consistent format.

## Why MCP?

- **Standardization**: Same interface for files, databases, APIs, web scraping
- **Reusability**: Pre-built servers for common tools (GitHub, Slack, Postgres, etc.)
- **Security**: Consistent authentication and permission model
- **Discovery**: LLMs can discover available tools dynamically

## How MCP Works

```
┌─────────────┐    MCP Protocol    ┌─────────────┐
│  LLM/Agent  │ ◄───────────────► │  MCP Server │ ◄──► External Service
└─────────────┘                    └─────────────┘
     Your code connects here            Pre-built or custom
```

1. **MCP Server** exposes tools and resources via a standard protocol
2. **Your code** connects to the server and discovers available capabilities
3. **LLM** receives tool definitions and can invoke them through your code

### Reference Card: MCP Concepts

| Concept | Description |
|:---|:---|
| **Server** | Process that exposes tools/resources (e.g., filesystem server, database server) |
| **Tool** | Function the LLM can invoke (e.g., `read_file`, `query_database`) |
| **Resource** | Data the LLM can read (e.g., file contents, API responses) |
| **Transport** | How client and server communicate (stdio, HTTP) |

### Code Snippet: Using MCP with OpenAI

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

client = OpenAI()

async def get_mcp_tools():
    """Connect to MCP server and get tool definitions."""
    # Connect to the filesystem MCP server
    server = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
    )
    
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools in OpenAI format
            mcp_tools = await session.list_tools()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in mcp_tools.tools
            ]

# Tools from MCP can be passed directly to OpenAI
# tools = asyncio.run(get_mcp_tools())
# response = client.chat.completions.create(model="gpt-4o", tools=tools, ...)
```

## Common MCP Servers

| Category | Server | Use Cases |
|:---|:---|:---|
| **File systems** | `@modelcontextprotocol/server-filesystem` | Read/write local files |
| **Databases** | `@modelcontextprotocol/server-postgres` | Query databases |
| **Web** | `@modelcontextprotocol/server-puppeteer` | Browser automation |
| **Code** | `@modelcontextprotocol/server-github` | Repository operations |

**Resources:**
- [MCP Documentation](https://modelcontextprotocol.io)
- [Pre-built servers](https://github.com/modelcontextprotocol/servers)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)

# Agentic LLMs

Moving beyond single request-response patterns, agentic LLMs can autonomously plan and execute multi-step tasks.

## Traditional vs Agentic LLM Use

| Traditional | Agentic |
|-------------|---------|
| Single request → single response | Multi-turn, self-guided iterations |
| User provides all context | Agent gathers information as needed |
| Fixed output | Iterates until task complete |
| No tool access | Can invoke external functions |

## Key Characteristics of Agents

- **Autonomy**: Agent decides next steps based on observations
- **Tool use**: Can invoke external functions (search, database queries, calculators)
- **Iteration**: Loops until task complete or max steps reached
- **State management**: Maintains context across multiple actions

## Example Agent Flow

```
Task: "Find recent papers on treatment X and summarize findings"
    ↓
1. Agent searches literature database (tool call)
    ↓
2. Agent reads top 3 papers (tool call)
    ↓
3. Agent synthesizes findings
    ↓
4. Agent checks if answer is complete
    ↓
   If not, searches for more specific info
    ↓
5. Returns final summary
```

### Reference Card: Agent Components

| Component | Purpose |
|:---|:---|
| **Planner** | Breaks task into steps |
| **Memory** | Stores conversation and results |
| **Tools** | External functions the agent can call |
| **Executor** | Runs tools and collects results |
| **Reflector** | Evaluates progress and adjusts |

### Code Snippet: Simple Agent Loop

```python
def agent_loop(task, tools, max_steps=10):
    """Simple agent loop with tool calling."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": task}
    ]
    
    for step in range(max_steps):
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # Check if done
        if message.tool_calls is None:
            return message.content
        
        # Execute tool calls
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call, tools)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return "Max steps reached"
```

## Prompting Techniques for Agents

![Agentic prompting patterns](media/agentic_prompting.png)

### Reference Card: Advanced Prompting Patterns

| Pattern | Description | Use Case |
|:---|:---|:---|
| **Chain-of-thought** | Make reasoning explicit step-by-step | Multi-step reasoning |
| **Self-consistency** | Generate multiple reasoning paths, vote | Improved accuracy |
| **ReAct** | Interleave reasoning and tool actions | Agent workflows |
| **Reflection** | Surface uncertainty and assumptions | Complex decisions |
| **Decision trees** | Explicit conditional logic | Structured workflows |

**Important Note**: LLM "reasoning" is not the same as thinking and does NOT always achieve better results or fewer hallucinations. It IS always more expensive. Use judiciously.

# Workflow Orchestration Patterns

Real tasks often span multiple steps and decision points. Workflows provide structure for complex LLM applications.

## Why Workflows?

- **Sequencing**: Chain LLM calls with conditional logic
- **State management**: Maintain context, handle partial failures
- **Tool integration**: Connect LLMs to databases, APIs, validation rules
- **Error handling**: Retries, fallbacks, human-in-the-loop checkpoints
- **Observability**: Track which step failed, inspect intermediate outputs

## Pattern: Prompt Chaining

**Concept**: Each LLM call processes output from previous call

**Benefits**: Each step is simple, testable, debuggable

### Code Snippet: Prompt Chain with OpenAI

```python
from openai import OpenAI

client = OpenAI()

def llm_call(prompt: str) -> str:
    """Simple wrapper for OpenAI chat completion."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_classify_summarize(document: str) -> dict:
    """Chain of LLM calls: extract → classify → summarize."""
    # Step 1: Extract entities
    entities = llm_call(f"Extract all medical entities from this text. Return as a list:\n{document}")
    
    # Step 2: Classify entities
    classified = llm_call(f"Classify these medical entities by type (condition, medication, procedure):\n{entities}")
    
    # Step 3: Generate summary
    summary = llm_call(f"Write a brief clinical summary based on:\n{classified}")
    
    return {"entities": entities, "classified": classified, "summary": summary}
```

## Pattern: Parallelization

**Concept**: Run independent LLM tasks simultaneously for speed

**Use cases**:

- **Divide-and-conquer**: Split document into sections, analyze in parallel
- **Multi-perspective**: Get different analyses of same content
- **Batch processing**: Process multiple items at once

> **Note**: `asyncio` is Python's built-in library for asynchronous programming. It lets you run multiple tasks concurrently without threads. `AsyncOpenAI` is the async version of the OpenAI client.

### Code Snippet: Parallel LLM Calls with asyncio

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def analyze_section(section: str, focus: str) -> dict:
    """Analyze one section with a specific focus."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Analyze this for {focus}:\n{section}"}]
    )
    return {"focus": focus, "analysis": response.choices[0].message.content}

async def parallel_analysis(document: str) -> list:
    """Analyze document from multiple perspectives in parallel."""
    tasks = [
        analyze_section(document, "diagnoses"),
        analyze_section(document, "medications"),
        analyze_section(document, "procedures"),
    ]
    results = await asyncio.gather(*tasks)
    return results

# Run with: results = asyncio.run(parallel_analysis(document))
```

## Pattern: Guardrails

**Concept**: Input/output monitors that enforce safety and compliance rules

### Reference Card: Common Guardrails

| Guardrail | Purpose |
|:---|:---|
| **PII/PHI detection** | Flag or redact Protected Health Information (PHI) or Personally Identifiable Information (PII) |
| **Hallucination detection** | Check if claims are grounded in source text |
| **Jailbreak detection** | Identify prompt injection attempts |
| **Format validation** | Ensure structured outputs meet schema |
| **Content filtering** | Block inappropriate content |

### Code Snippet: Input/Output Guardrails

```python
import re
from openai import OpenAI

client = OpenAI()

def detect_phi(text: str) -> dict | None:
    """Detect common PHI patterns in text."""
    patterns = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'mrn': r'\b(MRN|Medical Record)[\s:#]*\d+\b'
    }
    
    found = {}
    for phi_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[phi_type] = matches
    
    return found if found else None

def safe_llm_call(prompt: str) -> str:
    """LLM call with input and output guardrails."""
    # Input guardrail: check for PHI in prompt
    phi_in_prompt = detect_phi(prompt)
    if phi_in_prompt:
        raise ValueError(f"PHI detected in input: {phi_in_prompt.keys()}")
    
    # Make LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content
    
    # Output guardrail: check for PHI in response
    phi_in_output = detect_phi(output)
    if phi_in_output:
        raise ValueError(f"PHI detected in output: {phi_in_output.keys()}")
    
    return output
```

## Pattern: Deterministic Steps

**Concept**: Integrate rule-based logic alongside LLM calls

**Use cases where LLMs should NOT be used**:

- Dose calculations (use formulas)
- Date arithmetic
- Database lookups
- API calls with fixed parameters

```python
def process_patient_data(patient_info: str) -> dict:
    """Combine LLM analysis with deterministic calculations."""
    # LLM: Extract values from unstructured text
    extracted = llm_call(f"Extract weight_kg and height_m as JSON: {patient_info}")
    data = json.loads(extracted)
    
    # DETERMINISTIC: Calculate BMI (never trust LLM for math!)
    bmi = data['weight_kg'] / (data['height_m'] ** 2)
    
    # LLM: Generate interpretation
    interpretation = llm_call(f"Interpret BMI of {bmi:.1f} for this patient context")
    
    return {"bmi": round(bmi, 1), "interpretation": interpretation}
```

## Advanced Patterns (Further Reading)

For complex applications, additional patterns exist:

- **Orchestrator-Workers**: Central agent delegates to specialist workers
- **Evaluator-Optimizer**: Generate → evaluate → refine loops
- **Routing & Logic**: Conditional branching based on classification
- **Human-in-the-loop**: Pause for review before high-stakes actions

These are well-documented in the [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) and [OpenAI Cookbook](https://cookbook.openai.com/).

# LIVE DEMO!!!

Building an agentic workflow with tool calling, plus a brief look at MCP integration.

See: [demo/03_agentic_workflow.md](demo/03_agentic_workflow.md)


# Practical Recommendations

## Start Small

**"Baby" models** (low cost, quick):

- Mini/Nano tiers from major providers
- ~10x cheaper than flagship models
- Good for well-defined tasks

**Self-hosted options** (free, private):

- Ollama (desktop)
- PocketPal (iOS)
- No API costs, no usage limits
- Ideal for sensitive data prototyping

## Testing & Validation

**Start simple**:

- Test on 5-10 representative examples first
- Manually review outputs
- Try edge cases (missing data, unusual formats)
- Incorporate failures into few-shot examples

**Red flags to watch for**:

- Inconsistent outputs for similar inputs
- Made-up citations or facts
- Missing required information
- Wrong format or structure

**Remember**: Choose tasks that you can meaningfully oversee. Think of LLMs as prolific interns—productive but requiring supervision.

# Resources

## Prompt Engineering Guides

- **Anthropic**: [docs.anthropic.com/en/docs/build-with-claude/prompt-engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- **OpenAI**: [platform.openai.com/docs/examples](https://platform.openai.com/docs/examples)

## Workflow Tools

- **OpenAI Workflows**: [platform.openai.com/workflows](https://platform.openai.com/workflows)
- **LangChain**: [python.langchain.com/docs](https://python.langchain.com/docs)
- **LangGraph**: [langchain.com/langgraph](https://www.langchain.com/langgraph)

## Agent Frameworks

- **OpenAI Agents**: [platform.openai.com/docs/guides/agents](https://platform.openai.com/docs/guides/agents)
- **AutoGen**: [microsoft.github.io/autogen](https://microsoft.github.io/autogen)
- **smolagents**: [huggingface.co/docs/smolagents](https://huggingface.co/docs/smolagents)

## Self-Hosting

- **Ollama**: [ollama.com](https://ollama.com)
- **PocketPal**: [github.com/a-ghorbani/pocketpal-ai](https://github.com/a-ghorbani/pocketpal-ai)

## Academic Discounts

- **GitHub Education**: [github.com/education](https://github.com/education) - Free Pro with Copilot
- **ChatGPT for Teachers**: [openai.com/index/chatgpt-for-teachers](https://openai.com/index/chatgpt-for-teachers)
- **Claude for Education**: [claude.com/solutions/education](https://www.claude.com/solutions/education)
- **Gemini for Students**: [gemini.google/students](https://gemini.google/students/)
