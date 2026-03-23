
"""
PRACTICAL 9: Indian Language Sentiment Analysis with Translation & Multi-Model Approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
import warnings
import os
from collections import Counter
import nltk

warnings.filterwarnings('ignore')

# Download required NLP resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class IndianLanguageSentimentAnalyzer:

    def __init__(self):

        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        self.vader = SentimentIntensityAnalyzer()

        self.indian_languages = {
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi'
        }

        self.hindi_lexicon = {
            'अच्छा': 1.0, 'बुरा': -1.0,
            'सुंदर': 0.8, 'बेकार': -0.8,
            'खुश': 0.9, 'दुखी': -0.9,
            'प्यार': 0.9, 'नफरत': -0.9,
            'उत्कृष्ट': 1.0, 'भयानक': -1.0
        }

        self.emotion_keywords = {
            'joy': ['happy', 'glad', 'pleased', 'delighted'],
            'anger': ['angry', 'furious', 'mad', 'annoyed'],
            'sadness': ['sad', 'unhappy', 'depressed'],
            'fear': ['afraid', 'scared', 'terrified'],
        }

    # ------------------------------------------------

    def translate_to_english(self, text, source_lang='hi'):

        try:

            if source_lang == 'en':
                return text

            translated = GoogleTranslator(
                source=source_lang,
                target='en'
            ).translate(text)

            return translated

        except Exception as e:
            print("Translation error:", e)
            return None

    # ------------------------------------------------

    def vader_sentiment(self, text):

        scores = self.vader.polarity_scores(text)

        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, scores

    # ------------------------------------------------

    def textblob_sentiment(self, text):

        blob = TextBlob(text)

        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, polarity

    # ------------------------------------------------

    def detect_emotions(self, text):

        text_lower = text.lower()

        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(keyword in text_lower for keyword in keywords)
            emotion_scores[emotion] = score

        dominant = max(emotion_scores, key=emotion_scores.get)

        if emotion_scores[dominant] == 0:
            dominant = "neutral"

        return dominant

    # ------------------------------------------------

    def analyze_text(self, text, language):

        translated = self.translate_to_english(text, language)

        if not translated:
            return None

        vader_label, vader_scores = self.vader_sentiment(translated)
        blob_label, polarity = self.textblob_sentiment(translated)

        emotion = self.detect_emotions(translated)

        sentiments = [vader_label, blob_label]

        consensus = Counter(sentiments).most_common(1)[0][0]

        return {
            "original": text,
            "language": language,
            "translated": translated,
            "vader": vader_label,
            "textblob": blob_label,
            "polarity": polarity,
            "compound": vader_scores['compound'],
            "emotion": emotion,
            "consensus": consensus
        }

    # ------------------------------------------------

    def visualize_results(self, df):

        plt.figure(figsize=(10,6))

        sns.countplot(x="consensus", data=df,
                      palette={"Positive":"green","Negative":"red","Neutral":"gray"})

        plt.title("Sentiment Distribution")
        plt.savefig(f"{self.output_dir}/sentiment_chart.png")
        plt.close()

# ------------------------------------------------

def main():

    print("\nIndian Language Sentiment Analysis\n")

    analyzer = IndianLanguageSentimentAnalyzer()

    test_data = [

        ("This product is amazing", 'en'),
        ("Terrible service", 'en'),

        ("यह उत्पाद बहुत अच्छा है", 'hi'),
        ("बहुत बुरा अनुभव", 'hi'),

        ("मी खूप आनंदी आहे", 'mr'),
        ("हा चित्रपट फार वाईट आहे", 'mr')
    ]

    results = []

    for text, lang in test_data:

        result = analyzer.analyze_text(text, lang)

        if result:

            print("\nText:", text)
            print("Translated:", result["translated"])
            print("Sentiment:", result["consensus"])
            print("Emotion:", result["emotion"])

            results.append(result)

    df = pd.DataFrame(results)

    df.to_csv(f"{analyzer.output_dir}/sentiment_results.csv", index=False)

    analyzer.visualize_results(df)

    print(f"\nResults saved in {analyzer.output_dir} folder\n")


# ------------------------------------------------

if __name__ == "__main__":
    main()