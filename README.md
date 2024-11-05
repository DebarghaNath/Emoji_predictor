# Recurrent Neural Network Emoji Predictor

This project uses a Recurrent Neural Network (RNN) model to predict emojis based on input text, leveraging pre-trained GloVe embeddings and Keras's deep learning capabilities.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Project Overview
The **Emoji Predictor** is an RNN-based machine learning model that classifies text into emoji categories, representing the underlying sentiment or topic of the text with an emoji. This is achieved using a two-layer LSTM model trained on GloVe embeddings to capture semantic relationships in text.

## Dataset
The dataset used for training consists of a CSV file (`emoji_text.csv`) where:
- Column 0 (`X`) contains phrases or sentences.
- Column 1 (`Y`) contains numeric labels corresponding to emojis:
  - `0`: ❤️ (Red Heart)
  - `1`: ⚾ (Baseball)
  - `2`: 😀 (Grinning Face with Big Eyes)
  - `3`: 😞 (Disappointed Face)
  - `4`: 🍽️ (Fork and Knife with Plate)

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Emoji

## Installation
1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install tensorflow pandas numpy matplotlib emoji
