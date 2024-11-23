import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def read_spam():
    category = 'spam'
    directory = 'enron1\enron1\spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = 'enron1\enron1\ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df_ham = pd.DataFrame.from_records(ham)
df_spam = pd.DataFrame.from_records(spam)


df = pd.concat([df_ham, df_spam], ignore_index=True)


def preprocessor(e):
    e = re.sub('[^a-zA-Z]', ' ', e)
    return e.lower()
df['content'] = df['content'].apply(preprocessor)



vectorizer = CountVectorizer(preprocessor=preprocessor)

X_train, X_test, y_train, y_test = train_test_split(df['content'], df['category'], test_size=0.3, random_state=42)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

feature_names = vectorizer.get_feature_names_out()

importance = model.coef_[0]

top_positive_indices = importance.argsort()[-10:][::-1]
top_negative_indices = importance.argsort()[:10]

print("Top 10 spam words:")
for i in top_positive_indices:
    print(f"{feature_names[i]}: {importance[i]}")

print("\nTop 10 ham words:")
for i in top_negative_indices:
    print(f"{feature_names[i]}: {importance[i]}")
