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
```
---

## ⚙️ How to Use

### 1. Index the document collection

Run the following command to create the inverted index:

```bash
python3 index.py
```

This reads .txt files in the current directory 

### 2. Run the search engine

Run the query engine with:

```bash
python3 search.py
```

You will be prompted to:

- Enter a search query

- Choose output format: brief (docIDs only) or detailed (docID + matched lines)

- Accept or reject spelling correction suggestions if needed

### 🧪 Example Interaction

```bash
Enter your query: appl AND orangee
Did you mean: apple AND orange ?
Choose output mode: (1) Brief  (2) Detailed
> 2

docID: test2.txt
  "Fresh apple juice and orange slices are available."
```

### 🧠 NLP Techniques Used
| Technique             | Description                                             |
|-----------------------|---------------------------------------------------------|
| Tokenization          | Splits input into terms                                 |
| Stopword Removal      | Removes common words like "the", "is", etc.             |
| Stemming              | Applies Porter stemmer to reduce terms to root form     |
| Edit Distance         | Used for spelling correction                            |
| Boolean Query Parsing | Evaluates logic like AND, OR, NOT                       |

---

## 📚 Project Context
- 🎓 Course: COMP6714 – Information Retrieval and Web Search

- 🏫 University: UNSW Sydney

- 🧑‍💻 Term: T3 2024

- 💡 Project Type: Individual (Programming)

- 🎯 Focus: Applying NLP methods to classic search engine architecture

---

## 📜 License
This project is for academic learning purposes.

For personal or research adaptation, feel free to fork and build upon it. 🚀

---

## 🙋‍♀️ Author
Zhiqing (Bella) Pang

UNSW Master of Information Technology

📫 GitHub: @Bellapzq
