# System Architecture

## Team: ___________________
## Date: ___________________

---

## Architecture Diagram

> Replace this section with your team's architecture diagram.
> Export from FigJam, Miro, or draw.io and embed as an image,
> or describe the architecture in text with an ASCII diagram.

---

## Component Descriptions

### Corpus Layer
- **Source files:** `data/corpus/`
- **Chunking strategy:** *(describe your chunk size, overlap, and rationale)*
- **Metadata schema:** *(describe your metadata fields and why you chose them)*
- **Duplicate detection approach:** *(describe how chunk IDs are generated)*

### Vector Store Layer
- **Database:** ChromaDB (local persistent client)
- **Embedding model:** *(specify model name and provider)*
- **Similarity metric:** *(cosine / dot product)*
- **Retrieval k:** *(number of chunks retrieved per query)*
- **Similarity threshold:** *(minimum score to pass hallucination guard)*

### Agent Layer
- **Framework:** LangGraph
- **Graph nodes:** *(list your nodes and what each does)*
- **Conditional edges:** *(describe your routing logic)*
- **Memory:** *(describe how conversation history is managed)*

### Interface Layer
- **Framework:** *(Streamlit / Gradio)*
- **Deployment:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Key features:** *(list UI features implemented)*

---

## Design Decisions

> Document at least three deliberate design decisions your team made
> and the reasoning behind each. These are your interview talking points.

1. **Decision:** *(e.g. chunk size of 512 with 50 overlap)*
   **Rationale:** *(why this choice over alternatives)*

2. **Decision:**
   **Rationale:**

3. **Decision:**
   **Rationale:**

---

## Known Limitations

> Be honest about what your system does not handle well.
> Interviewers respect candidates who understand the limits of their own systems.

-
-
-

---

## What We Would Do With More Time

-
-
-
