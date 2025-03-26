# NextWordPredict

## Overview
NextWordPredict is a word prediction model that utilizes **Long Short-Term Memory (LSTM)** networks to predict the next word in a sequence. This project focuses on:
- **Sequence preprocessing**
- **Model tuning**
- **Evaluation for accurate text predictions**

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **LSTM**

## Dataset
The model is trained on a set of frequently asked questions (FAQs) from an educational course platform. The text data is tokenized and transformed into sequences for training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/NextWordPredict.git
   cd NextWordPredict
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy
   ```

## Model Architecture
The model consists of:
- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Captures contextual dependencies in the text.
- **Dense Layer**: Uses softmax activation for predicting the next word.

## Training the Model
The training pipeline includes:
1. **Tokenizing text**: Using Keras' `Tokenizer`.
2. **Creating input sequences**: Extracting n-grams from the text.
3. **Padding sequences**: Ensuring uniform input length.
4. **Building the LSTM model**: Training with categorical cross-entropy loss.
5. **Fitting the model**: Running for 100 epochs with Adam optimizer.

Run the training script:
```python
python train.py
```

## Predicting the Next Word
The model predicts the next word in a sequence based on prior words. Example usage:
```python
text = "total duration"
for i in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    pos = np.argmax(model.predict(padded_token_text))
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            print(text)
            break
```

## Results
The model generates reasonable predictions for next-word completion based on FAQ data. Improvements can be made by training on a larger, more diverse dataset.

## Future Work
- Implement **beam search** for better word predictions.
- Train on **larger datasets** for improved accuracy.
- Fine-tune using **pre-trained language models**.

## Contributors
- **Anshika Jain** (Developer)

## License
This project is licensed under the MIT License.
