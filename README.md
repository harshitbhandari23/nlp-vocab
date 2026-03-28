# Tokenizer and OOV Handling using TensorFlow Keras

## 📌 Project Overview

This project demonstrates how to use **TensorFlow Keras Tokenizer** for:

* Text Tokenization
* Word Index Creation
* Converting Text to Sequences
* Handling Unseen Words (Out-of-Vocabulary / OOV)

This is an important preprocessing step in **Natural Language Processing (NLP)** and **Deep Learning Models**.

---

## 🧠 Concepts Covered

* Tokenization
* Vocabulary creation
* Text to sequences
* Handling unseen words
* OOV Token

---

## 📦 Libraries Used

* TensorFlow / Keras

```python
from tensorflow.keras.preprocessing.text import Tokenizer
```

---

## 🚀 Example Code

### Step 1 — Basic Tokenization

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tk = Tokenizer()

corpus = ["cofee is hot", "Water is cold"]

tk.fit_on_texts(corpus)

print(tk.index_word)
```

### Output

Creates vocabulary index:

```
{
1: 'is',
2: 'cofee',
3: 'hot',
4: 'water',
5: 'cold'
}
```

---

## Step 2 — Convert Text to Sequences

```python
tk.texts_to_sequences(corpus)
```

### Output

```
[[2,1,3], [4,1,5]]
```

Each word is replaced by its index.

---

## Step 3 — Handling Unseen Words (OOV)

When new words appear that were not in training corpus.

```python
tk = Tokenizer(oov_token='<ovv>')

tk.fit_on_texts(corpus)

print(tk.index_word)
```

Now OOV token is added:

```
1 : <ovv>
2 : is
3 : cofee
4 : hot
5 : water
6 : cold
```

---

## Step 4 — New Corpus with Unseen Words

```python
corpus1 = ["black cofee is hot", "Pure water is cold"]

tk.texts_to_sequences(corpus1)
```

### Output

```
[[1,3,2,4], [1,5,2,6]]
```

Here:

* "black" → OOV → 1
* "Pure" → OOV → 1

---

## 📊 Why This is Important

Handling unseen words helps:

* Avoid model crashes
* Handle real-world text
* Improve generalization

Used in:

* Sentiment Analysis
* Fake News Detection
* Chatbots
* Text Classification

---

## 🎯 Use Cases

* NLP preprocessing
* Deep learning models
* LSTM / RNN models
* Word embeddings

---

## 👨‍💻 Author

Harshit Bhandari
Aspiring Data Analyst / NLP Enthusiast

---

## ⭐ If you found this helpful

Give this repo a ⭐
