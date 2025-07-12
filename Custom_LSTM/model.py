import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Downloading NLTK data...")
nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing utilities
class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = self.clean_text(text).split()
            word_counts.update(words)
        
        # Add special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        # Add words that appear at least min_freq times
        idx = 2
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def text_to_sequence(self, text, max_len=200):
        """Convert text to sequence of indices"""
        words = self.clean_text(text).split()
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence += [0] * (max_len - len(sequence))  # 0 is <PAD>
        
        return sequence

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_len=100):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        sequence = self.preprocessor.text_to_sequence(text, self.max_len)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Model architectures
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        # Use the last output
        last_output = rnn_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(RNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        lstm_out, (hidden, cell) = self.lstm(rnn_out)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class RNN_LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(RNN_LSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        lstm_out, (hidden, cell) = self.lstm(rnn_out)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.dropout(attended_output)
        output = self.fc(output)
        return output

# Training and evaluation functions
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the model and track metrics"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader):
    """Evaluate model and return predictions"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

# Generate synthetic sentiment data
def generate_synthetic_data(n_samples=5000):
    """Generate synthetic sentiment data for demonstration"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'brilliant', 'outstanding', 'superb']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'disappointing', 'pathetic', 'useless', 'annoying', 'frustrating']
    neutral_words = ['okay', 'fine', 'normal', 'average', 'typical', 'standard', 'regular', 'common', 'usual', 'ordinary']
    
    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'very', 'quite', 'rather', 'really', 'truly', 'actually', 'definitely', 'probably', 'possibly', 'maybe', 'perhaps']
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        # Randomly choose sentiment
        sentiment = np.random.choice([0, 1, 2])  # 0: negative, 1: neutral, 2: positive
        
        if sentiment == 0:  # Negative
            key_words = np.random.choice(negative_words, np.random.randint(1, 4))
        elif sentiment == 1:  # Neutral
            key_words = np.random.choice(neutral_words, np.random.randint(1, 3))
        else:  # Positive
            key_words = np.random.choice(positive_words, np.random.randint(1, 4))
        
        # Add some common words
        filler_words = np.random.choice(common_words, np.random.randint(3, 8))
        
        # Combine words
        all_words = list(key_words) + list(filler_words)
        np.random.shuffle(all_words)
        
        text = ' '.join(all_words)
        texts.append(text)
        labels.append(sentiment)
    
    return texts, labels

# Visualization functions
def plot_training_history(histories, model_names):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    axes[0, 0].set_title('Training Loss')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[0, 0].plot(history['train_losses'], label=name, marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation Loss
    axes[0, 1].set_title('Validation Loss')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[0, 1].plot(history['val_losses'], label=name, marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training Accuracy
    axes[1, 0].set_title('Training Accuracy')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[1, 0].plot(history['train_accuracies'], label=name, marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Accuracy
    axes[1, 1].set_title('Validation Accuracy')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[1, 1].plot(history['val_accuracies'], label=name, marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(predictions_list, targets_list, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    class_names = ['Negative', 'Neutral', 'Positive']
    
    for i, (pred, target, name) in enumerate(zip(predictions_list, targets_list, model_names)):
        cm = confusion_matrix(target, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        axes[i].set_title(f'{name} - Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def create_results_summary(histories, predictions_list, targets_list, model_names):
    """Create a summary of results"""
    results = []
    
    for i, (history, pred, target, name) in enumerate(zip(histories, predictions_list, targets_list, model_names)):
        final_train_acc = history['train_accuracies'][-1]
        final_val_acc = history['val_accuracies'][-1]
        test_acc = accuracy_score(target, pred) * 100
        
        results.append({
            'Model': name,
            'Final Train Accuracy': f'{final_train_acc:.2f}%',
            'Final Validation Accuracy': f'{final_val_acc:.2f}%',
            'Test Accuracy': f'{test_acc:.2f}%'
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return df

# Main execution
def main():
    print("Starting Sentiment Analysis Research")
    print("="*50)
    
    # Generate synthetic data
    print("Generating synthetic sentiment data...")
    texts, labels = generate_synthetic_data(n_samples=5000)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.4, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(X_train)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, preprocessor)
    val_dataset = SentimentDataset(X_val, y_val, preprocessor)
    test_dataset = SentimentDataset(X_test, y_test, preprocessor)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    vocab_size = preprocessor.vocab_size
    embedding_dim = 128
    hidden_dim = 64
    num_classes = 3
    num_epochs = 15
    
    # Initialize models
    models = {
        'Simple RNN': SimpleRNN(vocab_size, embedding_dim, hidden_dim, num_classes),
        'LSTM': LSTMModel(vocab_size, embedding_dim, hidden_dim, num_classes),
        'RNN+LSTM': RNN_LSTM(vocab_size, embedding_dim, hidden_dim, num_classes),
        'RNN+LSTM+Attention': RNN_LSTM_Attention(vocab_size, embedding_dim, hidden_dim, num_classes)
    }
    
    # Train and evaluate models
    histories = []
    predictions_list = []
    targets_list = []
    model_names = list(models.keys())
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)
        
        # Train model
        history = train_model(model, train_loader, val_loader, num_epochs)
        histories.append(history)
        
        # Evaluate on test set
        predictions, targets = evaluate_model(model, test_loader)
        predictions_list.append(predictions)
        targets_list.append(targets)
        
        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(targets, predictions, target_names=['Negative', 'Neutral', 'Positive']))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_history(histories, model_names)
    plot_confusion_matrices(predictions_list, targets_list, model_names)
    
    # Create results summary
    results_df = create_results_summary(histories, predictions_list, targets_list, model_names)
    
    print("\nResearch completed successfully!")
    print("All models have been trained and evaluated.")
    print("Check the visualizations for detailed performance comparison.")

if __name__ == "__main__":
    main()
