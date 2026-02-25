---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo 2: RAG Pipeline & MCP

Build a Retrieval-Augmented Generation pipeline that grounds LLM responses in actual clinical guidelines, then see how MCP standardizes tool integration.

## Learning Objectives

- Build a vector store from clinical documents using ChromaDB
- Implement the full RAG pipeline: embed → retrieve → augment → generate
- Compare RAG-grounded responses to direct LLM answers
- Understand how MCP standardizes tool discovery

## Setup

```python
%pip install -q sentence-transformers chromadb openai python-dotenv mcp
```

```python
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

load_dotenv()

# Configure client — works with OpenRouter or OpenAI
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = "openai/gpt-4o-mini"
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    MODEL = "gpt-4o-mini"
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

print(f"Using model: {MODEL}")
```

## Section 1: Clinical Knowledge Base

We'll work with synthetic clinical guideline chunks — the kind of documents a hospital might want an LLM to reference when answering clinical questions.

```python
clinical_knowledge = [
    {
        "id": "hypertension_1",
        "text": "Stage 1 hypertension is defined as systolic blood pressure 130-139 mmHg or diastolic 80-89 mmHg. Initial treatment includes lifestyle modifications: weight loss, DASH diet, sodium restriction (<2300mg/day), regular exercise (150 min/week moderate intensity).",
        "source": "JNC8 Guidelines"
    },
    {
        "id": "hypertension_2",
        "text": "First-line antihypertensive medications include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Choice depends on comorbidities: ACE inhibitors or ARBs preferred in diabetes and chronic kidney disease.",
        "source": "JNC8 Guidelines"
    },
    {
        "id": "diabetes_1",
        "text": "Type 2 diabetes is diagnosed with fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or 2-hour glucose ≥200 mg/dL during OGTT. Metformin is first-line therapy unless contraindicated. Target HbA1c <7% for most adults.",
        "source": "ADA Standards of Care"
    },
    {
        "id": "diabetes_2",
        "text": "For patients with type 2 diabetes and cardiovascular disease, SGLT2 inhibitors or GLP-1 receptor agonists with proven cardiovascular benefit are recommended regardless of HbA1c. Examples: empagliflozin, liraglutide.",
        "source": "ADA Standards of Care"
    },
    {
        "id": "chest_pain_1",
        "text": "Acute chest pain evaluation: HEART score assesses History, ECG, Age, Risk factors, and Troponin. Score 0-3 is low risk (discharge with follow-up), 4-6 is intermediate (admit for observation), 7-10 is high risk (early invasive strategy).",
        "source": "AHA Chest Pain Guidelines"
    },
    {
        "id": "chest_pain_2",
        "text": "STEMI diagnosis requires ST elevation ≥1mm in 2 contiguous leads or new LBBB with symptoms. Door-to-balloon time goal <90 minutes for primary PCI. If PCI not available within 120 minutes, fibrinolysis within 30 minutes.",
        "source": "AHA Chest Pain Guidelines"
    },
]

print(f"{len(clinical_knowledge)} guideline chunks loaded")
for doc in clinical_knowledge:
    print(f"  [{doc['source']}] {doc['text'][:60]}...")
```

## Section 2: Embedding & ChromaDB Indexing

An embedding is a fixed-length numeric vector that captures the *meaning* of a text chunk — similar documents end up near each other in vector space. ChromaDB is an in-memory vector database: we encode each chunk into an embedding, store them, and later retrieve the closest matches to a query.

```python
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="clinical_guidelines",
    metadata={"description": "Clinical practice guidelines"},
)

documents = [doc["text"] for doc in clinical_knowledge]
ids = [doc["id"] for doc in clinical_knowledge]
metadatas = [{"source": doc["source"]} for doc in clinical_knowledge]
embeddings = embedding_model.encode(documents).tolist()

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
    embeddings=embeddings,
)

print(f"Indexed {collection.count()} chunks in ChromaDB")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"\nFirst embedding (truncated): [{', '.join(f'{x:.4f}' for x in embeddings[0][:8])}, ...]")
```

## Section 3: RAG Query Function

The core RAG loop: embed the question → retrieve similar chunks → inject them as context → generate a grounded answer. The `n_results` parameter controls how many chunks to retrieve (more context = more information but also more noise and cost). The system prompt constrains the model to answer *only* from provided context — this is what makes RAG grounded rather than generative.

```python
def rag_query(question, n_results=3, show_sources=True):
    """Answer a question using retrieved guideline chunks as context."""
    # Step 1: Retrieve
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if show_sources:
        print("Retrieved documents:")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"  {i+1}. [{meta['source']}] (distance: {dist:.3f})")
            print(f"     {doc[:80]}...")
        print()

    # Step 2: Augment — build context from retrieved chunks
    context = "\n\n".join(results["documents"][0])

    # Step 3: Generate
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical assistant. Answer based ONLY on the provided context. "
                    "If the context doesn't contain enough information, say so. "
                    "Cite the guideline source when possible."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
```

```python
questions = [
    "What is the first-line treatment for hypertension in a patient with diabetes?",
    "How do I evaluate a patient with acute chest pain?",
    "What HbA1c level indicates diabetes?",
]

for q in questions:
    print("=" * 60)
    print(f"Q: {q}\n")
    answer = rag_query(q)
    print(f"A: {answer}\n")
```

Lower distance = higher semantic similarity. The model stays grounded because the system prompt constrains it to the retrieved context, and it can cite guideline sources because that metadata was stored alongside the embeddings.

## Section 4: RAG vs Direct LLM

What happens when the model answers *without* retrieved context? For well-known clinical facts (like hypertension thresholds) the LLM may already know the answer — but for organization-specific protocols, recent guideline updates, or internal policy, the model has no choice but to guess or refuse.

```python
def direct_llm_query(question):
    """Ask the LLM directly — no retrieval."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a clinical assistant. Be concise."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# Use a question where the guideline-specific detail matters:
# The HEART score thresholds (exact cutoffs) vary by institutional protocol.
# Our guidelines specify 0-3/4-6/7-10. The LLM may give different cutoffs.
test_q = (
    "According to the AHA Chest Pain Guidelines, what HEART score threshold "
    "separates low-risk from intermediate-risk patients, and what is the recommended "
    "action for each risk category?"
)

print("RAG Response (grounded in our guidelines):")
rag_answer = rag_query(test_q)
print(rag_answer)
print("\n" + "-" * 40 + "\n")
print("Direct LLM Response (from training data — may differ or hallucinate thresholds):")
print(direct_llm_query(test_q))
```

## Section 5: RAG with Citations

In clinical contexts, knowing *where* an answer came from is as important as the answer itself — a clinician needs to verify claims against the original guideline, not just trust the model. Numbering source chunks and instructing the model to cite them makes every claim traceable.

```python
def rag_with_citations(question, n_results=3):
    """RAG that returns answer + numbered source citations."""
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    context_parts = []
    sources = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        context_parts.append(f"[{i+1}] {doc}")
        sources.append(f"[{i+1}] {meta['source']}")

    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer based on the numbered sources. "
                    "Include citation numbers [1], [2], etc."
                ),
            },
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    return {"answer": response.choices[0].message.content, "sources": sources}


result = rag_with_citations(
    "What medications are recommended for diabetic patients with heart disease?"
)
print(f"Answer: {result['answer']}\n")
print("Sources:")
for s in result["sources"]:
    print(f"  {s}")
```

## Section 6: MCP — Standardized Tool Discovery and Use

We built our RAG pipeline by hand: custom embedding code, custom ChromaDB queries, custom prompt assembly. **Model Context Protocol (MCP)** standardizes all of this — pre-built servers expose tools that any LLM client can discover and call.

**Prerequisite**: Node.js and `npx` must be installed (`brew install node` on macOS, or [nodejs.org](https://nodejs.org)).

### Step 1: Discover tools from an MCP server

MCP uses async I/O — focus on the pattern (connect → discover → call), not the `async/await` details.

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the filesystem MCP server (points at current directory)
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "."],
)


async def discover_tools():
    """Connect to an MCP server and list available tools."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description[:80]}")
            return tools


mcp_tools = await discover_tools()
```

### Step 2: Call an MCP tool directly

The server handles execution — you just pass the tool name and arguments:

```python
async def call_mcp_tool(tool_name, arguments):
    """Call a tool on the MCP server and return the result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content[0].text


# List files in the current directory
files = await call_mcp_tool("list_directory", {"path": "."})
print(files)
```

```python
# Read a file through MCP
content = await call_mcp_tool("read_file", {"path": "sample_documents/guidelines.txt"})
print(content[:500])
```

### Step 3: Use MCP tools with an LLM agent

Convert MCP tool definitions to OpenAI format and plug them into an agent loop — the same loop from Demo 1:

```python
def mcp_to_openai_tools(mcp_tools):
    """Convert MCP tool definitions to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools.tools
    ]


openai_tools = mcp_to_openai_tools(mcp_tools)
print(f"Converted {len(openai_tools)} MCP tools to OpenAI format:")
for t in openai_tools:
    print(f"  {t['function']['name']}")
```

```python
import json


async def mcp_agent(task, max_steps=5):
    """Agent loop that uses MCP tools instead of hand-written functions."""
    messages = [
        {"role": "system", "content": "You can read and search files. Use the tools available to you."},
        {"role": "user", "content": task},
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for step in range(max_steps):
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=openai_tools
                )
                reply = response.choices[0].message
                messages.append(reply)

                if not reply.tool_calls:
                    return reply.content

                for tc in reply.tool_calls:
                    args = json.loads(tc.function.arguments)
                    print(f"  Step {step + 1}: calling {tc.function.name}({args})")

                    # MCP server executes the tool — not our code
                    result = await session.call_tool(tc.function.name, arguments=args)
                    tool_output = result.content[0].text

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    })

    return "Max steps reached"


# Ask the agent to find and summarize our clinical guidelines
answer = await mcp_agent("What files are in the sample_documents directory? Read the guidelines and summarize the key topics covered.")
print(answer)
```

The agent loop is the same as Demo 1 — but instead of hand-written `calculate_bmi()` functions, the tools come from an MCP server. The server handles execution; our code just passes messages.

| Manual Approach (Demo 1)                | MCP Approach                            |
| :-------------------------------------- | :-------------------------------------- |
| Define each tool function yourself      | Pre-built servers expose tools          |
| Write JSON schemas by hand              | Schemas auto-generated from server      |
| Your code executes functions            | MCP server executes functions           |
| Build each integration from scratch     | Plug-and-play servers                   |
| Great for learning                      | Great for production                    |

## Exercises

1. **Add more guidelines**: Add chunks about a new topic (e.g., anticoagulation, asthma management) and test retrieval
2. **Chunking experiment**: Split the longer guidelines into smaller chunks — does retrieval quality change?
3. **Hybrid search**: Filter by source metadata before doing similarity search
4. **Evaluation**: Write 5 questions with known answers and check if RAG gets them right
5. **MCP exploration**: Try connecting to a different MCP server (e.g., `@modelcontextprotocol/server-github`) and listing its tools

## Key Takeaways

- RAG grounds LLM responses in retrieved documents, reducing hallucination
- Vector databases (ChromaDB) enable fast semantic retrieval via embeddings
- Citation tracking makes RAG outputs verifiable
- MCP standardizes tool discovery _and execution_ — same agent loop, different tools
- The 3-step dance (model requests tool → server executes → result fed back) is the same whether tools are manual or MCP
