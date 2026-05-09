"""
Milestone III: Transformer-based Fake News Detection with Explainability
Student Proud Project - Machine Learning
Tajaddin Gafarov
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertModel,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Custom Dataset for BERT
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class FakeNewsTransformer:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
    def prepare_data(self, df, test_size=0.2):
        """Prepare data with title and text concatenation"""
        # Concatenate title and text for better context
        df['full_text'] = df['title'].fillna('') + ' [SEP] ' + df['text'].fillna('')
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['full_text'].values, 
            df['label'].values,
            test_size=test_size, 
            stratify=df['label'],
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5):
        """Fine-tune the transformer model"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        train_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            print(f'\n======== Epoch {epoch + 1} / {epochs} ========')
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_dataloader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            print(f'Average training loss: {avg_train_loss:.4f}')
            
            # Validation
            val_accuracy, val_f1, _, _ = self.evaluate(val_dataloader)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            print(f'Validation F1-Score: {val_f1:.4f}')
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_transformer_model.pt')
                print(f'New best model saved with F1: {best_val_f1:.4f}')
        
        return train_losses, val_accuracies, val_f1_scores
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        predictions = []
        actual_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
        
        accuracy = accuracy_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, average='weighted')
        roc_auc = roc_auc_score(actual_labels, probabilities)
        
        return accuracy, f1, roc_auc, (predictions, actual_labels, probabilities)
    
    def predict(self, texts):
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                predictions.append(pred.item())
                probabilities.append(probs[0, 1].item())
        
        return np.array(predictions), np.array(probabilities)
    
    def get_attention_weights(self, text):
        """Extract attention weights for interpretability"""
        self.model.eval()
        
        # Use DistilBertModel to get attention weights
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        bert_model.load_state_dict(self.model.distilbert.state_dict())
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Average attention weights across all layers and heads
        attentions = outputs.attentions
        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        avg_attention = torch.mean(avg_attention, dim=1)[0]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Create attention dictionary
        token_attention = {}
        for token, attention in zip(tokens, avg_attention.cpu().numpy()):
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                token_attention[token] = float(attention)
        
        return token_attention

def main():
    print("Loading FakeNewsNet dataset...")
    # Load the dataset
    df = pd.read_csv('data/processed/fakenewsnet_processed.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Initialize transformer model
    print("\nInitializing DistilBERT model...")
    transformer = FakeNewsTransformer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = transformer.prepare_data(df, test_size=0.2)
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = FakeNewsDataset(X_train, y_train, transformer.tokenizer)
    val_dataset = FakeNewsDataset(X_val, y_val, transformer.tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, transformer.tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    
    # Train the model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    train_losses, val_accs, val_f1s = transformer.train(
        train_dataloader, 
        val_dataloader, 
        epochs=3,
        learning_rate=2e-5
    )
    
    # Load best model for final evaluation
    transformer.model.load_state_dict(torch.load('best_transformer_model.pt'))
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    test_acc, test_f1, test_auc, (preds, labels, probs) = transformer.evaluate(test_dataloader)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'train_losses': train_losses,
        'val_accuracies': val_accs,
        'val_f1_scores': val_f1s,
        'predictions': preds,
        'labels': labels,
        'probabilities': probs
    }
    
    with open('transformer_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to transformer_results.pkl")
    
    # Example of attention visualization
    print("\n" + "="*50)
    print("Example Attention Analysis")
    print("="*50)
    
    sample_text = df.sample(1).iloc[0]['full_text']
    attention = transformer.get_attention_weights(sample_text[:200])
    
    # Show top 10 attended tokens
    sorted_attention = sorted(attention.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most attended tokens:")
    for token, weight in sorted_attention:
        print(f"  {token}: {weight:.4f}")
    
    return transformer, results

if __name__ == "__main__":
    transformer, results = main()
