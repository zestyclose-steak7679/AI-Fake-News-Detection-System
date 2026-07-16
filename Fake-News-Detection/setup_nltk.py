import nltk

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
