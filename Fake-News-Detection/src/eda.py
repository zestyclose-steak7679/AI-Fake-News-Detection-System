import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.logger import get_logger
from src.config import CLEANED_TRAIN_DATA_PATH, GRAPHS_DIR

logger = get_logger(__name__)

def generate_eda_graphs():
    """Generates 6 PNGs for EDA phase based on cleaned data."""
    logger.info("Starting EDA phase...")
    try:
        df = pd.read_csv(CLEANED_TRAIN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"File not found: {CLEANED_TRAIN_DATA_PATH}")
        raise e

    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label')
    plt.title('Class Distribution')
    plt.savefig(GRAPHS_DIR / 'class_distribution.png')
    plt.close()

    # 2. Missing Values
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(GRAPHS_DIR / 'missing_values.png')
    plt.close()

    # 3. Article Length
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(6, 4))
    sns.histplot(df['text_length'], bins=50)
    plt.title('Article Length Distribution')
    plt.savefig(GRAPHS_DIR / 'article_length.png')
    plt.close()

    # 4. Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='label', y='text_length')
    plt.title('Text Length by Class')
    plt.savefig(GRAPHS_DIR / 'boxplot.png')
    plt.close()

    # 5. Fake Wordcloud
    fake_text = " ".join(df[df['label'] == 1]['text'].astype(str))
    if fake_text.strip():
        wc_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title('Fake News Wordcloud')
        plt.savefig(GRAPHS_DIR / 'fake_wordcloud.png')
        plt.close()
    else:
        plt.figure()
        plt.savefig(GRAPHS_DIR / 'fake_wordcloud.png')
        plt.close()

    # 6. Real Wordcloud
    real_text = " ".join(df[df['label'] == 0]['text'].astype(str))
    if real_text.strip():
        wc_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_real, interpolation='bilinear')
        plt.axis('off')
        plt.title('Real News Wordcloud')
        plt.savefig(GRAPHS_DIR / 'real_wordcloud.png')
        plt.close()
    else:
        plt.figure()
        plt.savefig(GRAPHS_DIR / 'real_wordcloud.png')
        plt.close()

    logger.info("EDA phase completed successfully.")

if __name__ == "__main__":
    generate_eda_graphs()
