import pandas as pd
import re
import nltk
import time
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

tqdm.pandas()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def split_mbti_letters(df):
    df['I_E'] = df['type'].apply(lambda x: 'I' if x[0] == 'I' else 'E')
    df['N_S'] = df['type'].apply(lambda x: 'N' if x[1] == 'N' else 'S')
    df['T_F'] = df['type'].apply(lambda x: 'T' if x[2] == 'T' else 'F')
    df['J_P'] = df['type'].apply(lambda x: 'J' if x[3] == 'J' else 'P')
    return df

def load_data(path):
    print("📂 Loading dataset...")
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} rows.")

    print("🧹 Preprocessing text...")
    df['clean_posts'] = df['posts'].progress_apply(preprocess_text)

    print("🔡 Splitting MBTI types...")
    df = split_mbti_letters(df)

    return df

def train_classifiers(df):
    print("📊 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vect = vectorizer.fit_transform(df['clean_posts'])

    classifiers = {}
    labels = ['I_E', 'N_S', 'T_F', 'J_P']

    for label in labels:
        print(f"\n🔧 Training classifier for {label}...")
        y = df[label]
        X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, stratify=y, random_state=42)

        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"📈 Report for {label}:")
        print(classification_report(y_test, y_pred))

        classifiers[label] = model

    return vectorizer, classifiers

def predict_personality(text, vectorizer, classifiers):
    cleaned = preprocess_text(text)
    vect = vectorizer.transform([cleaned])
    result = ""
    result += classifiers['I_E'].predict(vect)[0]
    result += classifiers['N_S'].predict(vect)[0]
    result += classifiers['T_F'].predict(vect)[0]
    result += classifiers['J_P'].predict(vect)[0]
    return result

def main():
    df = load_data("trainingdata/mbti_1.csv")
    vectorizer, classifiers = train_classifiers(df)

    while True:
        try:
            example = input("\n🔮 Enter text to predict MBTI type (or 'exit' to quit): ")
            if example.lower() == 'exit':
                break
            mbti_type = predict_personality(example, vectorizer, classifiers)
            print(f"🧠 Predicted MBTI Type: {mbti_type}")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    main()
