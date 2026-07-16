import nltk
 fix-pipeline-implementation-12119849575071646015
import logging

logging.basicConfig(level=logging.INFO)

def download_nltk_resources():
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger_eng',
        'averaged_perceptron_tagger'
    ]
    for resource in resources:
        try:
            nltk.download(resource)
            logging.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logging.warning(f"Failed to download {resource}: {e}")

    print("NLTK setup completed.")

if __name__ == "__main__":
    download_nltk_resources()


resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
]

for resource in resources:
    try:
        nltk.download(resource)
    except Exception as e:
        print(f"Warning: Could not download {resource}: {e}")

print("NLTK setup completed.")
 implement-pipeline-16979291744340150157
