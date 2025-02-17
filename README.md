# 📚 Natural Language Processing (NLP) Cheat Sheet

## 📖 Table of Contents

- **Basic Terminology**
- **Text Preprocessing**
  - Tokenisation
  - Stopwords Removal
  - Stemming & Lemmatization
  - Text Normalisation
- **Text Representation**
  - Bag of Words (BoW)
  - TF-IDF
  - Word Embeddings
- **NLP Tasks**
  - Text Classification
  - Named Entity Recognition (NER)
  - Text Summarisation
  - Machine Translation
  - Sentiment Analysis
  - Question Answering
- **Language Models**
- **Evaluation Metrics**
- **Useful NLP Python Libraries**

---

## 🔹 Basic Terminology

| Term            | Definition |
|----------------|------------|
| **Corpus**      | A collection of text data used for training an NLP model. |
| **Token**       | A single unit of text, typically a word, but can be a character or a sentence. |
| **Vocabulary**  | A set of unique tokens from the text. |
| **n-gram**      | A contiguous sequence of n tokens (e.g., bigram = 2 tokens, trigram = 3 tokens). |
| **Stopwords**   | Common words (e.g., "and", "the") that are often removed in preprocessing. |

---

## 🔹 Text Preprocessing
Text preprocessing is the process of cleaning and preparing raw text data for analysis and modelling.

### **Tokenisation**
**Goal:** Split text into smaller units (tokens).

- **Word Tokenisation:** Split text into words.
  - **Example:** *"NLP is great!"* → `["NLP", "is", "great", "!"]`
- **Sentence Tokenisation:** Split text into sentences.
  - **Example:** *"Hello World. NLP is fun."* → `["Hello World.", "NLP is fun."]`
- **Libraries:** `nltk.word_tokenize()`, `spaCy.tokenizer`

### **Stopwords Removal**
**Goal:** Remove common words that do not contribute much meaning to the text.

- **Stopwords:** Words like *"is", "the", "in"*.
- **Libraries:** `nltk.corpus.stopwords`, `spaCy.stop_words`

### **Stemming & Lemmatization**
- **Stemming:** Reduces words to their root form by removing suffixes.
  - *Example:* "running" → "run"
- **Lemmatization:** Reduces words to their dictionary form, considering context.
  - *Example:* "better" → "good"
- **Libraries:** `nltk.stem.PorterStemmer`, `nltk.WordNetLemmatizer`, `spaCy`

### **Text Normalisation**
- **Lowercasing:** Convert all characters to lowercase to avoid case-sensitive mismatches.
- **Punctuation Removal:** Remove or ignore punctuation during processing.
- **Removing Numbers:** Often numbers are removed unless they carry meaningful information.
- **Libraries:** `re (regex)`, `nltk`, `spaCy`

---

## 🔹 Text Representation

### **Bag of Words (BoW)**
**Goal:** Represent text as a vector of word counts or frequencies, ignoring word order.

- Each word in the vocabulary is a feature.
- Creates sparse matrices (many 0 values for non-occurring words).
- **Library:** `sklearn.feature_extraction.text.CountVectorizer`

### **TF-IDF**
**Goal:** Weight words by their importance in the document and across the corpus.

- **TF (Term Frequency):** Frequency of a term in a document.
- **IDF (Inverse Document Frequency):** How rare a term is across the corpus.
- **Formula:** \( \text{TF-IDF} = \text{TF} \times \log(\frac{N}{DF}) \)
- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`

### **Word Embeddings**
**Goal:** Represent words as dense vectors that capture semantic meaning.

- **Word2Vec:** Predicts context words given a word (CBOW) or predicts a word given its context (Skip-gram).
- **GloVe:** Generates word vectors based on word co-occurrence in a corpus.
- **FastText:** Improves Word2Vec by considering subword information.
- **BERT Embeddings:** Contextual word embeddings from a pre-trained transformer model.
- **Libraries:** `gensim.models.Word2Vec`, `transformers`

---

## 🔹 NLP Tasks

### **Text Classification**
- **Goal:** Assign a label/category to a given piece of text (e.g., spam detection, sentiment analysis).
- **Algorithms:** Naïve Bayes, Logistic Regression, SVM, RNN, LSTM, Transformer-based models.

### **Named Entity Recognition (NER)**
- **Goal:** Identify and classify named entities in text into predefined categories (e.g., Person, Organisation, Location).
- **Example:** *"Barack Obama was born in Hawaii"* → `["Barack Obama" (PERSON), "Hawaii" (LOCATION)]`
- **Libraries:** `spaCy`, `transformers`

### **Text Summarisation**
- **Goal:** Generate a concise summary of a given text.
  - **Extractive Summarisation:** Extract key sentences from the original text.
  - **Abstractive Summarisation:** Generate a new summary that may include words not present in the original text.
- **Libraries:** `huggingface/transformers`

### **Sentiment Analysis**
- **Goal:** Determine the sentiment (positive, negative, neutral) of a piece of text.
- **Example:** *"The movie was amazing!"* → *Positive sentiment*
- **Libraries:** `TextBlob`, `VADER`, `transformers`

---

## 🔹 Useful NLP Python Libraries

```bash
pip install nltk spacy textblob gensim transformers flair vaderSentiment pattern
```

### **Example: Tokenisation with spaCy**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Natural Language Processing is fascinating!")
print([token.text for token in doc])
```

---

📌 **Author:** [@sarahsufi](https://github.com/sarahsufi)  



