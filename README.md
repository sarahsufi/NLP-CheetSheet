

# 📚 Natural Language Processing (NLP) Cheat Sheet

### Table of Contents
1. [Basic Terminology](#basic-terminology)
2. [Text Preprocessing](#text-preprocessing)
   - [Tokenization](#tokenization)
   - [Stopwords Removal](#stopwords-removal)
   - [Stemming & Lemmatization](#stemming--lemmatization)
   - [Text Normalization](#text-normalization)
3. [Text Representation](#text-representation)
   - [Bag of Words (BoW)](#bag-of-words-bow)
   - [TF-IDF](#tf-idf)
   - [Word Embeddings](#word-embeddings)
4. [NLP Tasks](#nlp-tasks)
   - [Text Classification](#text-classification)
   - [Named Entity Recognition (NER)](#named-entity-recognition-ner)
   - [Text Summarization](#text-summarization)
   - [Machine Translation](#machine-translation)
   - [Sentiment Analysis](#sentiment-analysis)
   - [Question Answering](#question-answering)
5. [Language Models](#language-models)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Useful NLP Python Libraries](#useful-nlp-python-libraries)

---

## Basic Terminology
- **Corpus**: A collection of text data used for training an NLP model.
- **Token**: A single unit of text, typically a word, but can be a character or a sentence.
- **Vocabulary**: A set of unique tokens from the text.
- **n-gram**: A contiguous sequence of `n` tokens (e.g., bigram = 2 tokens, trigram = 3 tokens).
- **Stopwords**: Common words (e.g., "and", "the") that are often removed in preprocessing.

---

## Text Preprocessing
Text preprocessing is the process of cleaning and preparing raw text data for analysis and modeling.

### Tokenization
**Goal**: Split text into smaller units (tokens).

- **Word Tokenization**: Split text into words.
  - Example: `"NLP is cool!" → ['NLP', 'is', 'cool', '!']`
  
- **Sentence Tokenization**: Split text into sentences.
  - Example: `"Hello World. NLP is fun." → ['Hello World.', 'NLP is fun.']`

Libraries: `nltk.word_tokenize()`, `spaCy.tokenizer`

### Stopwords Removal
**Goal**: Remove common words that don’t contribute much meaning to the text.
  
- **Stopwords**: Words like "is", "the", "in".
  
Library: `nltk.corpus.stopwords`, `spaCy.stop_words`

### Stemming & Lemmatization
**Stemming**: Reduce words to their base form (root/stem) by removing suffixes.
- Example: `"running" → "run"`

**Lemmatization**: Reduce words to their dictionary form (lemma), considering context.
- Example: `"better" → "good"`

Libraries: `nltk.stem.PorterStemmer`, `nltk.WordNetLemmatizer`, `spaCy`

### Text Normalization
- **Lowercasing**: Convert all characters to lowercase to avoid case-sensitive mismatches.
- **Punctuation Removal**: Remove or ignore punctuation during processing.
- **Removing Numbers**: Often numbers are removed unless they carry meaningful information.

Libraries: `re` (regex), `nltk`, `spaCy`

---

## Text Representation
### Bag of Words (BoW)
**Goal**: Represent text as a vector of word counts or frequencies, ignoring word order.

- Each word in the vocabulary is a feature.
- Creates sparse matrices (many 0 values for non-occurring words).
  
Library: `sklearn.feature_extraction.text.CountVectorizer`

### TF-IDF
**Goal**: Weight words by their importance in the document and across the corpus.

- **TF (Term Frequency)**: Frequency of a term in a document.
- **IDF (Inverse Document Frequency)**: How rare a term is across the corpus.
- Formula: \( \text{TF-IDF} = \text{TF} \times \log(\frac{N}{DF}) \)

Library: `sklearn.feature_extraction.text.TfidfVectorizer`

### Word Embeddings
**Goal**: Represent words as dense vectors that capture semantic meaning.

- **Word2Vec**: Predicts context words given a word (CBOW) or predicts a word given its context (Skip-gram).
- **GloVe**: Generates word vectors based on word co-occurrence in a corpus.
- **FastText**: Improves Word2Vec by considering subword information.
- **BERT Embeddings**: Contextual word embeddings from a pre-trained transformer model.

Libraries: `gensim.models.Word2Vec`, `transformers`

---

## NLP Tasks
### Text Classification
**Goal**: Assign a label/category to a given piece of text (e.g., spam detection, sentiment analysis).

- **Algorithms**:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Deep Learning Models (RNN, LSTM, Transformer-based models)

### Named Entity Recognition (NER)
**Goal**: Identify and classify named entities in text into predefined categories (e.g., Person, Organization, Location).

- **Example**: `"Barack Obama was born in Hawaii"` → `['Barack Obama' (PERSON), 'Hawaii' (LOCATION)]`

Libraries: `spaCy`, `transformers`

### Text Summarization
**Goal**: Generate a concise summary of a given text.

- **Extractive Summarization**: Extract key sentences from the original text.
- **Abstractive Summarization**: Generate a new summary that may include words not present in the original text.

Libraries: `huggingface/transformers`

### Machine Translation
**Goal**: Translate text from one language to another (e.g., English to French).

Libraries: `transformers`, `Google Translate API`

### Sentiment Analysis
**Goal**: Determine the sentiment (positive, negative, neutral) of a piece of text.

- **Example**: `"The movie was amazing!" → Positive sentiment`

Libraries: `TextBlob`, `VADER`, `transformers`

### Question Answering
**Goal**: Automatically answer questions based on a given context or corpus.

- **Example**: `"What is the capital of France?" → "Paris"`

Libraries: `huggingface/transformers`, `spaCy`

---

## Language Models
Language models predict the probability of a word sequence and generate text or complete tasks like translation, summarization, and QA.

- **n-gram Models**: Predict the next word based on the previous `n` words.
- **RNN (Recurrent Neural Networks)**: Handle sequential data but struggle with long-term dependencies.
- **LSTM (Long Short-Term Memory)**: Improved version of RNNs that better captures long-term dependencies.
- **GRU (Gated Recurrent Unit)**: A simplified version of LSTMs.
- **Transformers**: Use attention mechanisms to handle long-range dependencies in text (e.g., BERT, GPT, T5).

Libraries: `huggingface/transformers`, `tensorflow`, `torch`

---

## Evaluation Metrics
### Classification Metrics
- **Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision**: \( \frac{TP}{TP + FP} \)
- **Recall**: \( \frac{TP}{TP + FN} \)
- **F1-Score**: \( 2 \times \frac{Precision \times Recall}{Precision + Recall} \)
- **Confusion Matrix**: Shows TP, FP, TN, FN.

### Language Generation Metrics
- **BLEU (Bilingual Evaluation Understudy)**: Measures similarity between generated and reference text.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures overlap between generated and reference text.
- **Perplexity**: Measures how well a probability model predicts a sample.

---

## Useful NLP Python Libraries
- **NLTK**: Natural Language Toolkit (Preprocessing, Tokenization, POS tagging)
  - `pip install nltk`
- **spaCy**: Industrial-strength NLP with support for named entity recognition, dependency parsing.
  - `pip install spacy`
- **TextBlob**: Simplified text processing, sentiment analysis.
  - `pip install textblob`
- **Gensim**: Word2Vec, Topic Modeling, FastText, TF-IDF.
  - `pip install gensim`
- **Transformers (Hugging Face)**: Pretrained models for various NLP tasks (e.g., BERT, GPT, T5).
  - `pip install transformers`
- **Flair**: Simple framework for state-of-the-art NLP (NER, text classification).
  - `pip install flair`
- **VADER**: Sentiment analysis specific to social media text.
  - `pip install vaderSentiment`
- **Pattern**: Web mining module for Python (NLP, machine learning, network analysis).
  - `pip install pattern`

---



