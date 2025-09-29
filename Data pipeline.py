import pandas as pd
import re, string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from graphviz import Digraph

# ==========================
# 1. Setup NLTK
# ==========================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==========================
# 2. Load Dataset
# ==========================
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["id", "entity", "label", "tweet"]

# ==========================
# 3. Clean Text
# ==========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # remove mentions & hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df["clean_tweet"] = df["tweet"].astype(str).apply(clean_text)

# Encode labels
encoder = LabelEncoder()
df["label_encoded"] = encoder.fit_transform(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_tweet"], df["label_encoded"], test_size=0.2, random_state=42
)

# ==========================
# 4. Models
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = {}

for name, model in models.items():
    print(f"\n===== {name} =====")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Save accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    # Save model
    joblib.dump(pipeline, f"{name.replace(' ','_').lower()}_pipeline.pkl")

# ==========================
# 5. Bar Chart of Model Accuracies
# ==========================
plt.figure(figsize=(7,5))
plt.bar(results.keys(), results.values(), color=["skyblue", "lightgreen", "salmon"])
plt.ylabel("Accuracy")
plt.title("Model Comparison on Twitter Dataset")
plt.ylim(0,1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10, fontweight="bold")
plt.show()

# ==========================
# 6. Pipeline Diagram
# ==========================
pipeline_diagram = Digraph("DataPipeline", format="png")
pipeline_diagram.attr(size="6")

pipeline_diagram.node("A", "Raw Dataset\n(twitter_training.csv)", shape="box", style="filled", color="lightblue")
pipeline_diagram.node("B", "Data Cleaning\n(Stopwords, Lemma, Lowercase)", shape="box", style="filled", color="lightgreen")
pipeline_diagram.node("C", "Feature Extraction\n(TF-IDF, N-grams)", shape="box", style="filled", color="lightyellow")
pipeline_diagram.node("D", "Model Training & Evaluation\n(LogReg, NB, SVM)", shape="box", style="filled", color="lightpink")
pipeline_diagram.node("E", "Saved Model\n(.pkl)", shape="box", style="filled", color="orange")

pipeline_diagram.edges([("A","B"), ("B","C"), ("C","D"), ("D","E")])
pipeline_diagram.render("pipeline_diagram", view=True)

print("\n✅ Pipeline Completed! Models & Diagram Generated.")

