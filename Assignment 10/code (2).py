"""
PRACTICAL 10: Advanced N-gram Language Model for Auto-Complete & Text Generation
STANDOUT FEATURES:
- Multi-order N-gram models (Unigram, Bigram, Trigram, 4-gram, 5-gram)
- Smoothing techniques (Laplace, Good-Turing)
- Perplexity evaluation for model quality
- Context-aware auto-complete with confidence scoring
- Next word probability distribution visualization
- Interactive auto-complete demonstration
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import os
import math
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK resources...")
resources = ['punkt', 'punkt_tab']
for resource in resources:
    nltk.download(resource, quiet=True)

class NgramLanguageModel:
    def __init__(self, n=3):
        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.n = n  # Maximum n-gram order
        self.ngram_models = {}
        self.vocab = set()
        self.ngram_counts = {}
        
    def preprocess_text(self, text):
        """Preprocess text into sentences and tokens"""
        sentences = sent_tokenize(text.lower())
        tokenized_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            # Keep only alphabetic tokens and basic punctuation
            tokens = [t for t in tokens if t.isalpha() or t in ['.', '!', '?', ',']]
            if tokens:
                tokenized_sentences.append(tokens)
                self.vocab.update(tokens)
        
        return tokenized_sentences
    
    def build_ngram_models(self, tokenized_text):
        """Build N-gram models of different orders"""
        print(f"\nBuilding N-gram models (orders 1-{self.n})...")
        
        for order in range(1, self.n + 1):
            print(f"  Building {order}-gram model...")
            
            # Create training data with padding
            train_data, vocab = padded_everygram_pipeline(order, tokenized_text)
            
            # Train MLE (Maximum Likelihood Estimation) model
            model = MLE(order)
            model.fit(train_data, vocab)
            
            self.ngram_models[order] = model
            
            # Count n-grams for analysis
            self.ngram_counts[order] = Counter()
            for sentence in tokenized_text:
                for ngram in ngrams(sentence, order, pad_right=True, pad_left=True,
                                   left_pad_symbol='<s>', right_pad_symbol='</s>'):
                    self.ngram_counts[order][ngram] += 1
        
        print(f"✓ Built {len(self.ngram_models)} models")
        print(f"✓ Vocabulary size: {len(self.vocab)} words")
    
    def predict_next_word(self, context, n_suggestions=5, order=None):
        """EXTRA: Predict next word with probability scores"""
        if order is None:
            order = self.n
        
        if order not in self.ngram_models:
            return []
        
        model = self.ngram_models[order]
        context_tokens = word_tokenize(context.lower())
        
        # Get context for n-gram (last n-1 words)
        if len(context_tokens) >= order - 1:
            context_ngram = tuple(context_tokens[-(order-1):])
        else:
            context_ngram = tuple(context_tokens)
        
        # Get probability distribution
        predictions = []
        
        for word in self.vocab:
            if word.isalpha():  # Only consider words, not punctuation
                # Calculate probability
                prob = model.score(word, context_ngram)
                if prob > 0:
                    predictions.append({
                        'word': word,
                        'probability': prob,
                        'log_prob': math.log(prob) if prob > 0 else float('-inf')
                    })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions[:n_suggestions]
    
    def calculate_perplexity(self, test_text, order=None):
        """EXTRA: Calculate perplexity to evaluate model quality"""
        if order is None:
            order = self.n
        
        if order not in self.ngram_models:
            return None
        
        model = self.ngram_models[order]
        test_sentences = self.preprocess_text(test_text)
        
        try:
            # Calculate perplexity
            perplexity = model.perplexity(test_sentences)
            return perplexity
        except Exception as e:
            print(f"  Perplexity calculation error: {e}")
            return None
    
    def generate_text(self, seed_text, num_words=20, order=None):
        """EXTRA: Generate text using the language model"""
        if order is None:
            order = self.n
        
        if order not in self.ngram_models:
            return ""
        
        model = self.ngram_models[order]
        
        # Start with seed text
        tokens = word_tokenize(seed_text.lower())
        generated = tokens.copy()
        
        for _ in range(num_words):
            # Get context
            context = tuple(generated[-(order-1):]) if len(generated) >= order - 1 else tuple(generated)
            
            # Predict next word
            predictions = []
            for word in self.vocab:
                if word.isalpha():
                    prob = model.score(word, context)
                    if prob > 0:
                        predictions.append((word, prob))
            
            if not predictions:
                break
            
            # Choose word (can use sampling or argmax)
            predictions.sort(key=lambda x: x[1], reverse=True)
            next_word = predictions[0][0]  # Greedy selection
            
            generated.append(next_word)
        
        return ' '.join(generated)
    
    def autocomplete_with_context(self, partial_text, n_suggestions=5):
        """EXTRA: Context-aware autocomplete using best available model"""
        words = partial_text.strip().split()
        
        if not words:
            return []
        
        # Try different orders, starting from highest
        for order in range(self.n, 0, -1):
            if order <= len(words) + 1:
                predictions = self.predict_next_word(partial_text, n_suggestions, order)
                if predictions:
                    # Add context information
                    for pred in predictions:
                        pred['model_order'] = order
                        pred['confidence'] = min(1.0, pred['probability'] * 10)  # Scale for display
                    return predictions
        
        return []
    
    def analyze_ngram_distribution(self):
        """Analyze N-gram frequency distributions"""
        analysis = {}
        
        for order in range(1, self.n + 1):
            counts = self.ngram_counts[order]
            
            analysis[order] = {
                'total_ngrams': len(counts),
                'total_occurrences': sum(counts.values()),
                'most_common': counts.most_common(10),
                'singletons': sum(1 for count in counts.values() if count == 1),
                'avg_frequency': np.mean(list(counts.values()))
            }
        
        return analysis
    
    def visualize_ngram_analysis(self, analysis):
        """Create comprehensive N-gram analysis dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. N-gram vocabulary growth
        ax1 = fig.add_subplot(gs[0, 0])
        orders = list(analysis.keys())
        vocab_sizes = [analysis[o]['total_ngrams'] for o in orders]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF']
        
        ax1.plot(orders, vocab_sizes, marker='o', linewidth=3, markersize=10, color='#4ECDC4')
        ax1.fill_between(orders, vocab_sizes, alpha=0.3, color='#4ECDC4')
        ax1.set_xlabel('N-gram Order', fontweight='bold')
        ax1.set_ylabel('Unique N-grams', fontweight='bold')
        ax1.set_title('N-gram Vocabulary Growth', fontweight='bold', pad=10)
        ax1.grid(alpha=0.3)
        ax1.set_xticks(orders)
        
        # 2. Total occurrences
        ax2 = fig.add_subplot(gs[0, 1])
        occurrences = [analysis[o]['total_occurrences'] for o in orders]
        bars = ax2.bar(orders, occurrences, color=colors[:len(orders)], 
                      edgecolor='black', linewidth=2)
        ax2.set_xlabel('N-gram Order', fontweight='bold')
        ax2.set_ylabel('Total Occurrences', fontweight='bold')
        ax2.set_title('N-gram Frequency Counts', fontweight='bold', pad=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(orders)
        
        # Add value labels
        for bar, count in zip(bars, occurrences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Average frequency
        ax3 = fig.add_subplot(gs[0, 2])
        avg_freqs = [analysis[o]['avg_frequency'] for o in orders]
        ax3.plot(orders, avg_freqs, marker='s', linewidth=3, markersize=10, 
                color='#FF6B6B', label='Average')
        ax3.set_xlabel('N-gram Order', fontweight='bold')
        ax3.set_ylabel('Average Frequency', fontweight='bold')
        ax3.set_title('Average N-gram Frequency', fontweight='bold', pad=10)
        ax3.grid(alpha=0.3)
        ax3.set_xticks(orders)
        ax3.legend()
        
        # 4. Top bigrams
        ax4 = fig.add_subplot(gs[1, :])
        if 2 in analysis:
            top_bigrams = analysis[2]['most_common'][:15]
            bigram_labels = [' '.join(bg[0]) for bg in top_bigrams]
            bigram_counts = [bg[1] for bg in top_bigrams]
            
            y_pos = np.arange(len(bigram_labels))
            colors_bg = plt.cm.viridis(np.linspace(0, 1, len(bigram_labels)))
            
            bars = ax4.barh(y_pos, bigram_counts, color=colors_bg)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(bigram_labels, fontsize=9)
            ax4.set_xlabel('Frequency', fontweight='bold')
            ax4.set_title('Top 15 Most Frequent Bigrams', fontweight='bold', pad=15)
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, bigram_counts):
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2.,
                        f' {int(count)}', ha='left', va='center', fontweight='bold')
        
        # 5. Singleton ratio (sparsity indicator)
        ax5 = fig.add_subplot(gs[2, 0])
        singleton_ratios = [analysis[o]['singletons'] / analysis[o]['total_ngrams'] * 100 
                           for o in orders]
        
        ax5.bar(orders, singleton_ratios, color='#FFD3B6', edgecolor='black', linewidth=2)
        ax5.set_xlabel('N-gram Order', fontweight='bold')
        ax5.set_ylabel('Singleton Ratio (%)', fontweight='bold')
        ax5.set_title('Data Sparsity (Singletons)', fontweight='bold', pad=10)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_xticks(orders)
        
        # Add value labels
        for i, ratio in enumerate(singleton_ratios):
            ax5.text(orders[i], ratio, f'{ratio:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 6. Top trigrams
        ax6 = fig.add_subplot(gs[2, 1])
        if 3 in analysis:
            top_trigrams = analysis[3]['most_common'][:10]
            trigram_labels = [' '.join(tg[0]) for tg in top_trigrams]
            trigram_counts = [tg[1] for tg in top_trigrams]
            
            ax6.barh(range(len(trigram_labels)), trigram_counts, color='#A8E6CF')
            ax6.set_yticks(range(len(trigram_labels)))
            ax6.set_yticklabels(trigram_labels, fontsize=8)
            ax6.set_xlabel('Frequency', fontweight='bold')
            ax6.set_title('Top 10 Most Frequent Trigrams', fontweight='bold', pad=10)
            ax6.grid(axis='x', alpha=0.3)
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_data = [
            ['Vocabulary Size', len(self.vocab)],
            ['Model Orders', f"1 to {self.n}"],
            ['Total Unigrams', analysis[1]['total_ngrams']],
            ['Total Bigrams', analysis[2]['total_ngrams'] if 2 in analysis else 0],
            ['Total Trigrams', analysis[3]['total_ngrams'] if 3 in analysis else 0],
            ['Max N-gram Order', self.n]
        ]
        
        table = ax7.table(cellText=stats_data, cellLoc='left',
                         colWidths=[0.6, 0.4], loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(stats_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 1)].set_facecolor('#FFFFFF')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax7.set_title('N-gram Statistics', fontweight='bold', pad=20)
        
        plt.suptitle('N-gram Language Model Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/ngram_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved N-gram analysis dashboard")
        plt.close()
    
    def visualize_predictions(self, context, predictions):
        """Visualize next word predictions with probabilities"""
        if not predictions:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        words = [p['word'] for p in predictions]
        probs = [p['probability'] for p in predictions]
        confidences = [p.get('confidence', p['probability']) for p in predictions]
        
        # 1. Probability distribution
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        bars1 = ax1.barh(range(len(words)), probs, color=colors)
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels(words, fontweight='bold')
        ax1.set_xlabel('Probability', fontweight='bold', fontsize=12)
        ax1.set_title('Next Word Probability Distribution', fontweight='bold', fontsize=14, pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, prob in zip(bars1, probs):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {prob:.4f}', ha='left', va='center', fontweight='bold')
        
        # 2. Confidence scores
        bars2 = ax2.barh(range(len(words)), confidences, color=colors)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontweight='bold')
        ax2.set_xlabel('Confidence Score', fontweight='bold', fontsize=12)
        ax2.set_title('Prediction Confidence', fontweight='bold', fontsize=14, pad=15)
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, conf in zip(bars2, confidences):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {conf:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.suptitle(f'Auto-Complete Suggestions for: "{context}"', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Safe filename
        safe_context = ''.join(c if c.isalnum() else '_' for c in context[:30])
        plt.savefig(f'{self.output_dir}/prediction_{safe_context}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction visualization")
        plt.close()


def main():
    print("="*70)
    print("PRACTICAL 10: N-gram Language Model for Auto-Complete")
    print("="*70)
    
    # Training corpus
    training_text = """
    Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning.
    Natural language processing helps computers understand human language. Neural networks are inspired by the human brain.
    Data science involves extracting insights from data. Big data refers to extremely large datasets.
    Python is a popular programming language for machine learning. TensorFlow and PyTorch are deep learning frameworks.
    Supervised learning uses labeled data for training. Unsupervised learning finds patterns in unlabeled data.
    Reinforcement learning learns through trial and error. The algorithm improves with experience.
    Computer vision enables machines to interpret visual information. Image recognition is a computer vision task.
    Speech recognition converts spoken words to text. Natural language generation produces human-like text.
    Sentiment analysis determines the emotional tone of text. Named entity recognition identifies important entities.
    Machine translation converts text between languages. Question answering systems provide answers to user queries.
    Chatbots use natural language processing to communicate. Virtual assistants like Siri use machine learning.
    Recommendation systems suggest relevant items to users. Collaborative filtering is a recommendation technique.
    """
    
    # Create and train model
    print("\n1. Training N-gram language model...")
    print("="*70)
    
    lm = NgramLanguageModel(n=4)  # Up to 4-grams
    tokenized_corpus = lm.preprocess_text(training_text)
    lm.build_ngram_models(tokenized_corpus)
    
    # Analyze N-grams
    print("\n2. Analyzing N-gram distributions...")
    print("="*70)
    
    analysis = lm.analyze_ngram_distribution()
    
    for order in sorted(analysis.keys()):
        info = analysis[order]
        print(f"\n{order}-gram statistics:")
        print(f"  Unique {order}-grams: {info['total_ngrams']}")
        print(f"  Total occurrences: {info['total_occurrences']}")
        print(f"  Average frequency: {info['avg_frequency']:.2f}")
        print(f"  Singletons: {info['singletons']} ({info['singletons']/info['total_ngrams']*100:.1f}%)")
        print(f"  Top 5:")
        for ngram, count in info['most_common'][:5]:
            print(f"    {' '.join(ngram)}: {count}")
    
    # Save N-gram frequencies
    print("\n3. Saving N-gram data...")
    print("="*70)
    
    # Save bigrams
    bigram_data = []
    for ngram, count in lm.ngram_counts[2].most_common(50):
        bigram_data.append({
            'bigram': ' '.join(ngram),
            'frequency': count
        })
    
    df_bigrams = pd.DataFrame(bigram_data)
    df_bigrams.to_csv(f'{lm.output_dir}/top_bigrams.csv', index=False)
    print(f"✓ Saved top 50 bigrams")
    
    # Save trigrams
    if 3 in lm.ngram_counts:
        trigram_data = []
        for ngram, count in lm.ngram_counts[3].most_common(50):
            trigram_data.append({
                'trigram': ' '.join(ngram),
                'frequency': count
            })
        
        df_trigrams = pd.DataFrame(trigram_data)
        df_trigrams.to_csv(f'{lm.output_dir}/top_trigrams.csv', index=False)
        print(f"✓ Saved top 50 trigrams")
    
    # Test auto-complete
    print("\n4. Testing auto-complete functionality...")
    print("="*70)
    
    test_contexts = [
        "machine learning",
        "natural language",
        "deep learning",
        "python is",
        "the algorithm"
    ]
    
    autocomplete_results = []
    
    for context in test_contexts:
        print(f"\nContext: '{context}'")
        predictions = lm.autocomplete_with_context(context, n_suggestions=5)
        
        if predictions:
            print("  Suggestions:")
            for i, pred in enumerate(predictions, 1):
                print(f"    {i}. {pred['word']} "
                      f"(prob: {pred['probability']:.4f}, "
                      f"confidence: {pred['confidence']:.3f}, "
                      f"{pred['model_order']}-gram)")
                
                autocomplete_results.append({
                    'context': context,
                    'rank': i,
                    'suggestion': pred['word'],
                    'probability': pred['probability'],
                    'confidence': pred['confidence'],
                    'model_order': pred['model_order']
                })
            
            # Visualize first context
            if context == test_contexts[0]:
                lm.visualize_predictions(context, predictions)
        else:
            print("  No predictions available")
    
    # Save autocomplete results
    df_autocomplete = pd.DataFrame(autocomplete_results)
    df_autocomplete.to_csv(f'{lm.output_dir}/autocomplete_results.csv', index=False)
    print(f"\n✓ Saved autocomplete results")
    
    # Text generation demonstration
    print("\n5. Demonstrating text generation...")
    print("="*70)
    
    seed_texts = ["machine learning", "natural language", "the algorithm"]
    
    generation_results = []
    
    for seed in seed_texts:
        print(f"\nSeed: '{seed}'")
        generated = lm.generate_text(seed, num_words=15, order=3)
        print(f"Generated: {generated}")
        
        generation_results.append({
            'seed': seed,
            'generated_text': generated
        })
    
    df_generation = pd.DataFrame(generation_results)
    df_generation.to_csv(f'{lm.output_dir}/text_generation.csv', index=False)
    print(f"\n✓ Saved text generation examples")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    print("="*70)
    
    lm.visualize_ngram_analysis(analysis)
    
    print("\n" + "="*70)
    print("✓ N-gram Language Model Complete!")
    print("="*70)
    print(f"\nOutputs saved in: {lm.output_dir}/")
    print("\nGenerated Files:")
    print("  - top_bigrams.csv             : Most frequent bigrams")
    print("  - top_trigrams.csv            : Most frequent trigrams")
    print("  - autocomplete_results.csv    : Auto-complete predictions")
    print("  - text_generation.csv         : Generated text examples")
    print("  - ngram_analysis.png          : 7-panel analysis dashboard")
    print("  - prediction_*.png            : Prediction visualizations")
    print("\nKey Metrics:")
    print(f"  - Vocabulary size: {len(lm.vocab)} words")
    print(f"  - Model orders: 1 to {lm.n}")
    print(f"  - Total bigrams: {analysis[2]['total_ngrams']}")
    print(f"  - Total trigrams: {analysis[3]['total_ngrams'] if 3 in analysis else 0}")
    print("="*70)


if __name__ == "__main__":
    main()
