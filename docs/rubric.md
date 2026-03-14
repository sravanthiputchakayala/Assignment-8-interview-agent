# Hour 3 Competition Rubric
## Deep Learning RAG Interview Prep Agent

---

## Judging Instructions

Each team is scored by their **peer teams** during Hour 3 plus instructor
override. Use this sheet to score the presenting team. Be honest — you
are preparing each other for real interviews.

**Scoring team:** ___________________
**Presenting team:** ___________________
**Date:** ___________________

---

## Scoring Categories

### 1. It Works (20 points)

The live demo runs without crashing. Ingestion, retrieval, and chat all
function end to end.

| Criteria | Points Available | Score |
|---|---|---|
| App launches and loads without errors | 5 | |
| Document ingestion completes successfully | 5 | |
| Chat returns a relevant response with source citation | 5 | |
| Duplicate ingestion is detected and reported | 5 | |
| **Subtotal** | **20** | |

**Notes:**
_______________________________________________

---

### 2. Architecture Clarity (20 points)

The team can draw and explain the full pipeline from corpus to response
without hesitation. Every team member owns the whole system.

| Criteria | Points Available | Score |
|---|---|---|
| Can explain how chunks flow from file to ChromaDB | 5 | |
| Can explain how LangChain retrieves from the vector store | 5 | |
| Can explain what LangGraph nodes do and why | 5 | |
| Diagram committed to `docs/architecture.md` | 5 | |
| **Subtotal** | **20** | |

**Notes:**
_______________________________________________

---

### 3. Corpus Quality (15 points)

Content is accurate, well-chunked, covers required topics, and includes
metadata and landmark papers.

| Criteria | Points Available | Score |
|---|---|---|
| All 6 core topics present (ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder) | 5 | |
| At least one landmark paper per core topic ingested | 5 | |
| Chunks follow the schema — metadata complete and accurate | 5 | |
| **Subtotal** | **15** | |

**Notes:**
_______________________________________________

---

### 4. Code Quality (15 points)

OOP structure, PEP 8 compliance, no hardcoded keys, readable and organized.

| Criteria | Points Available | Score |
|---|---|---|
| Classes used throughout — no monolithic scripts | 5 | |
| PEP 8 followed — naming, spacing, docstrings present | 5 | |
| No hardcoded API keys — environment variables used | 5 | |
| **Subtotal** | **15** | |

**Notes:**
_______________________________________________

---

### 5. Interview Performance (20 points)

Can each team member answer a technical question about any part of the
system — not just their own role.

| Criteria | Points Available | Score |
|---|---|---|
| First team member called on answers correctly | 5 | |
| Second team member called on answers correctly | 5 | |
| Answers cite specific technical decisions, not vague descriptions | 5 | |
| No "that was the engineer's part" deflections | 5 | |
| **Subtotal** | **20** | |

> **Automatic deduction:** -5 points if any team member says
> "that was [role]'s part" and cannot answer.

**Notes:**
_______________________________________________

---

### 6. Bonus Topics (5 points)

SOM, Boltzmann Machine, or GAN content present, ingested, and retrievable.

| Criteria | Points Available | Score |
|---|---|---|
| At least one bonus topic in corpus and retrievable via chat | 3 | |
| All three bonus topics present | 2 | |
| **Subtotal** | **5** | |

**Notes:**
_______________________________________________

---

### 7. Wow Factor (5 points)

Something unexpected — a clever LangGraph loop, a smart UI feature,
a particularly sharp corpus, streaming responses, hybrid search,
re-ranking, async ingestion, or anything that signals production maturity.

| Criteria | Points Available | Score |
|---|---|---|
| One genuinely impressive feature beyond the spec | 3 | |
| Team can explain the technical reasoning behind it | 2 | |
| **Subtotal** | **5** | |

**What impressed you:**
_______________________________________________

---

## Final Score

| Category | Max | Score |
|---|---|---|
| It Works | 20 | |
| Architecture Clarity | 20 | |
| Corpus Quality | 15 | |
| Code Quality | 15 | |
| Interview Performance | 20 | |
| Bonus Topics | 5 | |
| Wow Factor | 5 | |
| **TOTAL** | **100** | |

---

## Peer Feedback (required)

**One thing this team did really well:**

_______________________________________________

**One thing this team should improve before their real interview:**

_______________________________________________

**Would you hire this team based on what you saw today?**

☐ Yes, immediately   ☐ Yes, with minor reservations   ☐ Not yet
