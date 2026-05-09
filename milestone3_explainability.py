"""
Explainability Module for Fake News Detection
Implements LIME and SHAP for model interpretability (Fixed Attention Logic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import shap
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')

class FakeNewsExplainer:
    """
    Unified explainability class for both traditional ML and transformer models
    """
    
    def __init__(self, model_type='transformer', model=None):
        self.model_type = model_type
        self.model = model
        self.explainer = None
        
        if model_type == 'transformer':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
    def init_lime(self, class_names=['Real', 'Fake']):
        """Initialize LIME explainer"""
        self.lime_explainer = LimeTextExplainer(
            class_names=class_names,
            verbose=False
        )
    
    def transformer_predict_proba(self, texts):
        """Prediction function for transformer models"""
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        all_probs = []
        
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
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs[0])
        
        return np.array(all_probs)
    
    def explain_with_lime(self, text, num_features=10):
        """Generate LIME explanation for a single prediction"""
        if self.model_type == 'transformer':
            predict_fn = self.transformer_predict_proba
        else:
            predict_fn = self.model.predict_proba
        
        exp = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=100
        )
        
        # Get the explanation as a list of (word, importance) tuples
        explanation = exp.as_list()
        
        # Get prediction probabilities
        probs = predict_fn([text])[0]
        prediction = 'Fake' if probs[1] > 0.5 else 'Real'
        confidence = max(probs)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {'Real': probs[0], 'Fake': probs[1]},
            'explanation': explanation,
            'lime_object': exp
        }
    
    def create_explanation_report(self, text, title="Explanation Report"):
        """Create a comprehensive explanation report combining multiple methods"""
        report = {
            'text': text[:500] + '...' if len(text) > 500 else text,
            'title': title
        }
        
        # Get LIME explanation
        lime_exp = self.explain_with_lime(text)
        report['prediction'] = lime_exp['prediction']
        report['confidence'] = lime_exp['confidence']
        report['probabilities'] = lime_exp['probabilities']
        report['lime_features'] = lime_exp['explanation']
        
        # For transformer models, also get attention weights
        if self.model_type == 'transformer':
            try:
                report['attention'] = self.get_attention_analysis(text)
            except Exception as e:
                print(f"Warning: Attention analysis failed: {e}")
                report['attention'] = []
        
        return report
    
    def get_attention_analysis(self, text):
        """Analyze attention patterns in transformer model (Fixed Dimensions)"""
        self.model.eval()
        
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
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention weights tuple (layers, batch, heads, seq, seq)
        attentions = outputs.attentions
        
        # 1. Stack and average across layers -> (batch, heads, seq, seq)
        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        
        # 2. Average across heads -> (batch, seq, seq)
        avg_attention = torch.mean(avg_attention, dim=1)
        
        # 3. Get first batch item -> (seq, seq)
        matrix = avg_attention[0]
        
        # 4. Collapse to 1D: Average attention RECEIVED by each token (column mean)
        # This tells us which tokens were "looked at" the most by other tokens
        token_importance = torch.mean(matrix, dim=0)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Find important tokens
        important_tokens = []
        # Calculate threshold on CPU
        threshold = token_importance.mean().item()
        token_scores = token_importance.cpu().numpy()
        
        for idx, (token, attn) in enumerate(zip(tokens, token_scores)):
            # Ignore special tokens
            if token not in ['[PAD]', '[CLS]', '[SEP]'] and attn > threshold:
                important_tokens.append({
                    'token': token,
                    'attention': float(attn),
                    'position': idx
                })
        
        # Sort by attention weight
        important_tokens = sorted(important_tokens, key=lambda x: x['attention'], reverse=True)[:20]
        
        return important_tokens
    
    def visualize_explanation(self, report, save_path=None):
        """Create visualization of the explanation report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Explanation Report: {report['title']}", fontsize=16)
        
        # 1. Prediction Probabilities
        ax1 = axes[0, 0]
        classes = list(report['probabilities'].keys())
        probs = list(report['probabilities'].values())
        bars = ax1.bar(classes, probs, color=['green', 'red'])
        ax1.set_ylabel('Probability')
        ax1.set_title(f"Prediction: {report['prediction']} (Confidence: {report['confidence']:.3f})")
        ax1.set_ylim([0, 1])
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # 2. LIME Feature Importance
        ax2 = axes[0, 1]
        if 'lime_features' in report:
            features = report['lime_features'][:10]
            words = [f[0] for f in features]
            scores = [f[1] for f in features]
            colors = ['green' if s > 0 else 'red' for s in scores]
            
            ax2.barh(range(len(words)), scores, color=colors)
            ax2.set_yticks(range(len(words)))
            ax2.set_yticklabels(words)
            ax2.set_xlabel('LIME Score')
            ax2.set_title('Top Contributing Words (LIME)')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Text Sample
        ax3 = axes[1, 0]
        ax3.axis('off')
        text_sample = report['text'][:300] + '...'
        ax3.text(0.05, 0.95, f"Text Sample:\n\n{text_sample}", 
                transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', wrap=True)
        
        # 4. Attention Weights
        ax4 = axes[1, 1]
        if 'attention' in report and report['attention']:
            tokens = report['attention'][:10]
            token_words = [t['token'] for t in tokens]
            token_weights = [t['attention'] for t in tokens]
            
            ax4.barh(range(len(token_words)), token_weights, color='blue')
            ax4.set_yticks(range(len(token_words)))
            ax4.set_yticklabels(token_words)
            ax4.set_xlabel('Attention Weight')
            ax4.set_title('Top Attention Tokens')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, 'Attention Analysis\nNot Available', 
                    transform=ax4.transAxes, fontsize=12, 
                    ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        return fig
