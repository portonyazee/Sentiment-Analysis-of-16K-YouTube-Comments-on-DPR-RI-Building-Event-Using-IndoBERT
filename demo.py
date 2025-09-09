# Sentiment Analysis with IndoBERT + Preprocessing + Hasil Tabel

import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Load dataset
df = pd.read_csv("demodpr25agustus2025.csv") 
print("Sampel data:")
display(df.head())

# 2. Preprocessing
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()                                  # case folding
    text = re.sub(r"http\S+|www.\S+", " ", text)              # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                  # hapus angka & simbol
    text = re.sub(r"\s+", " ", text).strip()                  # hapus spasi berlebih
    tokens = [w for w in text.split() if w not in stop_words] # hapus stopwords
    text = " ".join(tokens)
    text = stemmer.stem(text)                                 # stemming
    return text

df["clean_comment"] = df["comment"].apply(clean_text)
print("\nHasil preprocessing:")
display(df[["comment", "clean_comment"]].head())

# 3. Load model IndoBERT Sentiment
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=model_name,
    truncation=True,
    max_length=512
)

# 4. Prediksi sentimen (dengan skor probabilitas)
sentiments = []
scores = []

for text in df["clean_comment"]:
    if text.strip() != "":
        result = sentiment_pipeline(str(text))[0]
        sentiments.append(result["label"])
        scores.append(result["score"])
    else:
        sentiments.append(None)
        scores.append(None)


df["sentiment"] = sentiments
df["confidence"] = scores

print("\nHasil prediksi sentimen per komentar:")
display(df[["comment", "clean_comment", "sentiment", "confidence"]].head(10))

# 5. Visualisasi distribusi sentimen
sentiment_counts = df["sentiment"].value_counts()

# Pie chart
plt.figure(figsize=(6,6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Distribusi Sentimen")
plt.show()

# Bar chart
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Jumlah Komentar per Sentimen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Komentar")
plt.show()

# 6. Wordcloud untuk tiap sentimen
for sentiment in df["sentiment"].unique():
    text = " ".join(df[df["sentiment"] == sentiment]["clean_comment"].astype(str))
    if text.strip() != "":
        wc = WordCloud(width=800, height=400, background_color="white",
                       stopwords=stop_words, collocations=False).generate(text)

        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud untuk Sentimen: {sentiment}")
        plt.show()

# 7. Simpan hasil analisis
df.to_csv("hasil_sentimen.csv", index=False)
df.to_excel("tabel_hasil_sentimen.xlsx", index=False)

print("\n Hasil sentimen disimpan ke 'hasil_sentimen.csv' dan 'tabel_hasil_sentimen.xlsx'")
