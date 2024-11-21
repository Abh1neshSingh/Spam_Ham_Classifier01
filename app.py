from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load pre-trained model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
ps = PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    text = [word for word in text if word.isalnum()]  # Remove non-alphanumeric characters
    text = [word for word in text if word not in stop_words and word not in string.punctuation]  # Remove stopwords/punctuation
    text = [ps.stem(word) for word in text]  # Apply stemming
    return " ".join(text)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("message")  # Fetch input from the form
        if not input_text:
            return render_template("index.html", result="No input provided!", message=None)
        
        # Process and classify the input
        transformed_text = transform_text(input_text)
        print("Transformed Text:", transformed_text)  # Debugging: Check preprocessing
        vectorized_text = vectorizer.transform([transformed_text])
        print("Vectorized Text:", vectorized_text)  # Debugging: Check vectorization
        prediction = model.predict(vectorized_text)[0]
        print("Prediction:", prediction)  # Debugging: Check prediction
        
        # Map prediction to result
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", result=result, message=input_text)

    # Default case for GET request
    return render_template("index.html", result=None, message=None)

if __name__ == "__main__":
    app.run(debug=True)
