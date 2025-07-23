# 🧠 NLP-Enhanced Information Retrieval System

This is a lightweight search engine implemented in Python for COMP6714: Information Retrieval and Web Search at UNSW.  
It supports ranked retrieval, Boolean search, and spelling correction using traditional NLP techniques — **no external libraries** or machine learning frameworks required.

> 🔍 Built from scratch using concepts like inverted indexing, TF-IDF scoring, and edit distance for real-word and non-word spelling corrections.

---

## 🚀 Features

- ✅ **Inverted Index with Positional Information**  
  Enables fast document lookup and term position matching.

- ✅ **TF-IDF Ranked Retrieval**  
  Returns most relevant documents sorted by term frequency-inverse document frequency.

- ✅ **Boolean Search Support**  
  Handles queries like `apple AND orange`, `car OR bus`, `NOT truck`.

- ✅ **Spelling Correction**  
  Supports:
  - **Non-word errors** (e.g., `tigerz` → `tigers`)
  - **Real-word errors** (e.g., `form` vs `from`)

- ✅ **Dual Output Modes**  
  - Brief: Only `docID`s  
  - Detailed: `docID` + matching line content

---

## 🗂️ File Structure

```bash
.
├── index.py          # Build the inverted index from a collection of .txt documents
├── search.py         # Query engine: handles preprocessing, ranking, and spelling correction
├── test1.txt         # Example document
├── test2.txt         # Example document
├── MyTestIndex/      # Stores generated index files
