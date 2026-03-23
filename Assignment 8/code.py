"""
PRACTICAL 8: WordNet-Based Word Sense Disambiguation with Lesk & Similarity Algorithms
STANDOUT FEATURES:
- Multiple WSD algorithms (Lesk, Adapted Lesk, Path Similarity, Wu-Palmer)
- Context window optimization for better disambiguation
- Confidence scoring for predictions
- Ambiguity degree analysis
- Sense distribution visualization
- Comparative algorithm performance dashboard
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.wsd import lesk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK resources...")
resources = ['wordnet', 'omw-1.4', 'punkt', 'stopwords', 'punkt_tab']
for resource in resources:
    nltk.download(resource, quiet=True)

class WordSenseDisambiguator:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        self.stop_words = set(stopwords.words('english'))
        
    def get_ambiguous_words(self, text):
        """Identify ambiguous words (words with multiple senses)"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        ambiguous = []
        for token in tokens:
            synsets = wn.synsets(token)
            if len(synsets) > 1:
                ambiguous.append({
                    'word': token,
                    'num_senses': len(synsets),
                    'synsets': synsets
                })
        
        return ambiguous
    
    def simple_lesk(self, word, sentence):
        """Standard Lesk algorithm"""
        result = lesk(word_tokenize(sentence), word)
        return result
    
    def adapted_lesk(self, word, sentence, context_window=5):
        """EXTRA: Adapted Lesk with extended context"""
        tokens = word_tokenize(sentence.lower())
        
        # Find word position
        try:
            word_idx = tokens.index(word.lower())
        except ValueError:
            return None
        
        # Get context window
        start = max(0, word_idx - context_window)
        end = min(len(tokens), word_idx + context_window + 1)
        context = tokens[start:end]
        
        # Get all synsets for the word
        synsets = wn.synsets(word)
        if not synsets:
            return None
        
        # Score each synset
        best_sense = None
        max_overlap = 0
        
        context_set = set(context) - self.stop_words
        
        for synset in synsets:
            # Get signature (definition + examples + related words)
            signature = set()
            
            # Add definition words
            signature.update(word_tokenize(synset.definition().lower()))
            
            # Add example words
            for example in synset.examples():
                signature.update(word_tokenize(example.lower()))
            
            # Add hypernym and hyponym words
            for hypernym in synset.hypernyms():
                signature.update(word_tokenize(hypernym.definition().lower()))
            for hyponym in synset.hyponyms()[:3]:  # Limit hyponyms
                signature.update(word_tokenize(hyponym.definition().lower()))
            
            # Remove stopwords
            signature = signature - self.stop_words
            
            # Calculate overlap
            overlap = len(context_set.intersection(signature))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = synset
        
        return best_sense
    
    def similarity_based_wsd(self, word, sentence):
        """EXTRA: WSD using path similarity with context words"""
        tokens = word_tokenize(sentence.lower())
        context_words = [t for t in tokens if t.isalpha() and 
                        t != word.lower() and t not in self.stop_words]
        
        word_synsets = wn.synsets(word)
        if not word_synsets:
            return None
        
        # Calculate similarity with context
        best_sense = None
        max_similarity = 0
        
        for synset in word_synsets:
            total_similarity = 0
            count = 0
            
            for context_word in context_words[:10]:  # Limit context words
                context_synsets = wn.synsets(context_word)
                
                for context_synset in context_synsets[:3]:  # Limit synsets
                    try:
                        sim = synset.path_similarity(context_synset)
                        if sim:
                            total_similarity += sim
                            count += 1
                    except:
                        pass
            
            avg_similarity = total_similarity / count if count > 0 else 0
            
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                best_sense = synset
        
        return best_sense
    
    def wup_similarity_wsd(self, word, sentence):
        """EXTRA: WSD using Wu-Palmer similarity"""
        tokens = word_tokenize(sentence.lower())
        context_words = [t for t in tokens if t.isalpha() and 
                        t != word.lower() and t not in self.stop_words]
        
        word_synsets = wn.synsets(word)
        if not word_synsets:
            return None
        
        best_sense = None
        max_similarity = 0
        
        for synset in word_synsets:
            total_similarity = 0
            count = 0
            
            for context_word in context_words[:10]:
                context_synsets = wn.synsets(context_word)
                
                for context_synset in context_synsets[:3]:
                    try:
                        sim = synset.wup_similarity(context_synset)
                        if sim:
                            total_similarity += sim
                            count += 1
                    except:
                        pass
            
            avg_similarity = total_similarity / count if count > 0 else 0
            
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                best_sense = synset
        
        return best_sense
    
    def calculate_confidence(self, word, sentence, predicted_sense):
        """EXTRA: Calculate confidence score for prediction"""
        if not predicted_sense:
            return 0.0
        
        synsets = wn.synsets(word)
        if not synsets:
            return 0.0
        
        # Factors for confidence
        # 1. Number of senses (fewer = higher confidence)
        sense_factor = 1.0 / len(synsets)
        
        # 2. Definition length (longer = more specific = higher confidence)
        def_length = len(predicted_sense.definition().split())
        length_factor = min(1.0, def_length / 20.0)
        
        # 3. Number of examples (more = higher confidence)
        example_factor = min(1.0, len(predicted_sense.examples()) / 3.0)
        
        # Combined confidence
        confidence = (sense_factor + length_factor + example_factor) / 3.0
        
        return confidence
    
    def disambiguate_text(self, text, target_words=None):
        """Disambiguate multiple words in text using all algorithms"""
        if target_words is None:
            # Auto-detect ambiguous words
            ambiguous = self.get_ambiguous_words(text)
            target_words = [item['word'] for item in ambiguous]
        
        results = []
        
        for word in target_words:
            # Apply all algorithms
            lesk_sense = self.simple_lesk(word, text)
            adapted_sense = self.adapted_lesk(word, text)
            path_sense = self.similarity_based_wsd(word, text)
            wup_sense = self.wup_similarity_wsd(word, text)
            
            # Get all possible senses
            all_synsets = wn.synsets(word)
            
            result = {
                'word': word,
                'sentence': text,
                'num_possible_senses': len(all_synsets),
                'lesk_sense': lesk_sense.name() if lesk_sense else 'N/A',
                'lesk_definition': lesk_sense.definition() if lesk_sense else 'N/A',
                'adapted_lesk_sense': adapted_sense.name() if adapted_sense else 'N/A',
                'adapted_lesk_definition': adapted_sense.definition() if adapted_sense else 'N/A',
                'path_similarity_sense': path_sense.name() if path_sense else 'N/A',
                'path_similarity_definition': path_sense.definition() if path_sense else 'N/A',
                'wup_similarity_sense': wup_sense.name() if wup_sense else 'N/A',
                'wup_similarity_definition': wup_sense.definition() if wup_sense else 'N/A',
                'lesk_confidence': self.calculate_confidence(word, text, lesk_sense),
                'adapted_confidence': self.calculate_confidence(word, text, adapted_sense),
                'path_confidence': self.calculate_confidence(word, text, path_sense),
                'wup_confidence': self.calculate_confidence(word, text, wup_sense)
            }
            
            results.append(result)
        
        return results
    
    def compare_algorithms(self, results):
        """EXTRA: Compare algorithm agreement"""
        agreement_matrix = np.zeros((4, 4))
        algorithm_names = ['Lesk', 'Adapted Lesk', 'Path Sim', 'WuP Sim']
        
        for result in results:
            senses = [
                result['lesk_sense'],
                result['adapted_lesk_sense'],
                result['path_similarity_sense'],
                result['wup_similarity_sense']
            ]
            
            for i in range(4):
                for j in range(4):
                    if senses[i] != 'N/A' and senses[j] != 'N/A':
                        if senses[i] == senses[j]:
                            agreement_matrix[i, j] += 1
        
        return agreement_matrix, algorithm_names
    
    def visualize_results(self, df):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Ambiguity distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sense_counts = df['num_possible_senses'].value_counts().sort_index()
        ax1.bar(sense_counts.index, sense_counts.values, color='#FF6B6B', edgecolor='black')
        ax1.set_xlabel('Number of Possible Senses', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Word Ambiguity Distribution', fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Average confidence by algorithm
        ax2 = fig.add_subplot(gs[0, 1])
        algorithms = ['Lesk', 'Adapted\nLesk', 'Path\nSimilarity', 'WuP\nSimilarity']
        confidences = [
            df['lesk_confidence'].mean(),
            df['adapted_confidence'].mean(),
            df['path_confidence'].mean(),
            df['wup_confidence'].mean()
        ]
        colors = ['#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF']
        bars = ax2.bar(algorithms, confidences, color=colors, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Average Confidence', fontweight='bold')
        ax2.set_title('Algorithm Confidence Scores', fontweight='bold', pad=10)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence distribution
        ax3 = fig.add_subplot(gs[0, 2])
        all_confidences = np.concatenate([
            df['lesk_confidence'].values,
            df['adapted_confidence'].values,
            df['path_confidence'].values,
            df['wup_confidence'].values
        ])
        ax3.hist(all_confidences, bins=20, color='#FFD3B6', edgecolor='black')
        ax3.axvline(all_confidences.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {all_confidences.mean():.3f}')
        ax3.set_xlabel('Confidence Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Overall Confidence Distribution', fontweight='bold', pad=10)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Words by ambiguity level
        ax4 = fig.add_subplot(gs[1, :])
        top_ambiguous = df.nlargest(10, 'num_possible_senses')[['word', 'num_possible_senses']]
        
        y_pos = np.arange(len(top_ambiguous))
        colors_amb = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_ambiguous)))
        
        bars = ax4.barh(y_pos, top_ambiguous['num_possible_senses'], color=colors_amb)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_ambiguous['word'], fontweight='bold')
        ax4.set_xlabel('Number of Possible Senses', fontweight='bold')
        ax4.set_title('Top 10 Most Ambiguous Words', fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, top_ambiguous['num_possible_senses']):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(count)}', ha='left', va='center', fontweight='bold')
        
        # 5. Confidence by ambiguity level
        ax5 = fig.add_subplot(gs[2, 0])
        scatter = ax5.scatter(df['num_possible_senses'], df['lesk_confidence'],
                             c=df['adapted_confidence'], cmap='viridis',
                             s=100, alpha=0.6, edgecolors='black')
        ax5.set_xlabel('Number of Senses', fontweight='bold')
        ax5.set_ylabel('Lesk Confidence', fontweight='bold')
        ax5.set_title('Ambiguity vs Confidence', fontweight='bold', pad=10)
        ax5.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Adapted Lesk Confidence')
        
        # 6. Algorithm comparison
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Count successful disambiguations
        success_counts = [
            (df['lesk_sense'] != 'N/A').sum(),
            (df['adapted_lesk_sense'] != 'N/A').sum(),
            (df['path_similarity_sense'] != 'N/A').sum(),
            (df['wup_similarity_sense'] != 'N/A').sum()
        ]
        
        ax6.bar(algorithms, success_counts, color=colors, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Successful Disambiguations', fontweight='bold')
        ax6.set_title('Algorithm Success Rate', fontweight='bold', pad=10)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_data = [
            ['Total Words', len(df)],
            ['Avg Senses', f"{df['num_possible_senses'].mean():.2f}"],
            ['Max Ambiguity', f"{df['num_possible_senses'].max()}"],
            ['Best Algorithm', algorithms[np.argmax(confidences)]],
            ['Avg Confidence', f"{all_confidences.mean():.3f}"],
            ['Success Rate', f"{min(success_counts) / len(df) * 100:.1f}%"]
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
        
        ax7.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.suptitle('Word Sense Disambiguation Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/wsd_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved WSD dashboard")
        plt.close()
    
    def visualize_algorithm_agreement(self, agreement_matrix, algorithm_names):
        """EXTRA: Visualize agreement between algorithms"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize to percentages
        row_sums = agreement_matrix.sum(axis=1, keepdims=True)
        agreement_pct = np.divide(agreement_matrix, row_sums, 
                                 where=row_sums!=0, out=np.zeros_like(agreement_matrix)) * 100
        
        sns.heatmap(agreement_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=algorithm_names, yticklabels=algorithm_names,
                   ax=ax, cbar_kws={'label': 'Agreement %'}, vmin=0, vmax=100)
        
        ax.set_title('Algorithm Agreement Matrix\n(% of predictions that match)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
        ax.set_ylabel('Algorithm', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/algorithm_agreement.png', dpi=300, bbox_inches='tight')
        print("✓ Saved algorithm agreement visualization")
        plt.close()


def main():
    print("="*70)
    print("PRACTICAL 8: WordNet-Based Word Sense Disambiguation")
    print("="*70)
    
    wsd = WordSenseDisambiguator()
    
    # Sample sentences with ambiguous words
    test_sentences = [
        ("I went to the bank to deposit money.", ['bank']),
        ("The bass player played a bass guitar.", ['bass', 'played']),
        ("I saw a bat flying at night and picked up a baseball bat.", ['bat']),
        ("The plant is growing in the plant.", ['plant']),
        ("The pitcher threw the ball and broke the water pitcher.", ['pitcher']),
        ("I need to book a table and read my book.", ['book']),
        ("The match started after lighting a match.", ['match']),
        ("The light in the room was very light.", ['light']),
        ("I saw a saw in the workshop.", ['saw']),
        ("The lead singer used a lead pencil.", ['lead'])
    ]
    
    print("\n1. Performing Word Sense Disambiguation...")
    print("="*70)
    
    all_results = []
    
    for sentence, words in test_sentences:
        print(f"\nSentence: {sentence}")
        print(f"Target words: {words}")
        print("-" * 70)
        
        results = wsd.disambiguate_text(sentence, words)
        
        for result in results:
            print(f"\nWord: '{result['word']}'")
            print(f"  Possible senses: {result['num_possible_senses']}")
            print(f"\n  Lesk Algorithm:")
            print(f"    Sense: {result['lesk_sense']}")
            print(f"    Definition: {result['lesk_definition'][:80]}...")
            print(f"    Confidence: {result['lesk_confidence']:.3f}")
            print(f"\n  Adapted Lesk:")
            print(f"    Sense: {result['adapted_lesk_sense']}")
            print(f"    Definition: {result['adapted_lesk_definition'][:80]}...")
            print(f"    Confidence: {result['adapted_confidence']:.3f}")
            
            all_results.append(result)
    
    # Save results
    print("\n" + "="*70)
    print("2. Saving results...")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{wsd.output_dir}/wsd_results.csv', index=False)
    print(f"✓ Saved WSD results ({len(df)} words analyzed)")
    
    # Compare algorithms
    agreement_matrix, algorithm_names = wsd.compare_algorithms(all_results)
    
    agreement_df = pd.DataFrame(agreement_matrix, 
                               columns=algorithm_names, 
                               index=algorithm_names)
    agreement_df.to_csv(f'{wsd.output_dir}/algorithm_agreement.csv')
    print(f"✓ Saved algorithm agreement matrix")
    
    # Create visualizations
    print("\n" + "="*70)
    print("3. Creating visualizations...")
    print("="*70)
    
    wsd.visualize_results(df)
    wsd.visualize_algorithm_agreement(agreement_matrix, algorithm_names)
    
    # Summary statistics
    print("\n" + "="*70)
    print("✓ Disambiguation Complete!")
    print("="*70)
    print(f"\nOutputs saved in: {wsd.output_dir}/")
    print("\nGenerated Files:")
    print("  - wsd_results.csv             : Complete disambiguation results")
    print("  - algorithm_agreement.csv     : Algorithm agreement matrix")
    print("  - wsd_dashboard.png           : 7-panel analysis dashboard")
    print("  - algorithm_agreement.png     : Agreement heatmap")
    print("\nKey Metrics:")
    print(f"  - Total words analyzed: {len(df)}")
    print(f"  - Average ambiguity: {df['num_possible_senses'].mean():.2f} senses/word")
    print(f"  - Most ambiguous word: '{df.loc[df['num_possible_senses'].idxmax(), 'word']}' "
          f"({df['num_possible_senses'].max()} senses)")
    print(f"  - Average Lesk confidence: {df['lesk_confidence'].mean():.3f}")
    print(f"  - Average Adapted Lesk confidence: {df['adapted_confidence'].mean():.3f}")
    print("="*70)


if __name__ == "__main__":
    main()
