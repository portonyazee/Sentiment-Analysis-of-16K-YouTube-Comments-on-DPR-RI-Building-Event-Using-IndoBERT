# Sentiment Analysis of 16K YouTube Comments on DPR RI Building Event Using IndoBERT

## Project Overview
This project focuses on **sentiment analysis of 16,000 YouTube comments** related to the **DPR RI Building event (August 25, 2025)**. The model leverages **IndoBERT**, a pre-trained BERT model for the Indonesian language, to classify public sentiment into categories such as **positive, negative, and neutral**.

The goal is to understand **public opinion** and provide **insightful visualizations** for further analysis.

## Features
- Data preprocessing of raw YouTube comments
- Sentiment classification using IndoBERT
- Evaluation metrics (accuracy, precision, recall, F1-score)
- Visualization of sentiment distribution
- Reproducible workflow for research and analysis

## Dataset
- **Source**: YouTube comments from videos related to the DPR RI Building event (August 25, 2025)
- **Total Records**: ~16,000 comments
- **Format**: CSV (comment text + label)

## Tech Stack
- **Python 3.x**
- **Transformers (Hugging Face)**
- **PyTorch**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
- **Model**: IndoBERT
- **Best Accuracy**: XX% (replace with actual result)
- **Visualization**: Distribution of positive, negative, and neutral sentiments


## Acknowledgements
- IndoBERT (https://huggingface.co/indobenchmark/indobert-base-p1)
- Hugging Face Transformers (https://huggingface.co/transformers/)
- YouTube API for comment extraction
