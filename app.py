from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "hackathon_secret_key"  # required for sessions

# ---------------- LOAD MODEL ----------------
pipeline = joblib.load('models/fake_news_model.pkl')
tfidf = pipeline.named_steps['tfidf']
model = pipeline.named_steps['model']


# ---------------- SPLASH PAGE ----------------
@app.route('/')
def splash():
    return render_template('splash.html')


# ---------------- LOGIN PAGE ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Demo credentials (perfect for hackathon)
        if email == "mythri@gmail.com" and password == "123456":
            session['user'] = email
            return redirect(url_for('home'))
        else:
            error = "Invalid email or password"

    return render_template('login.html', error=error)


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ---------------- MAIN APP ----------------
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


# ---------------- PREDICTION API ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    text = data.get('text', '').strip()
    unverified = data.get('unverified', False)

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Prediction
    prediction = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    confidence = max(proba) * 100

    # Ethical confidence adjustment
    if unverified:
        confidence = max(confidence - 10, 0)

    label = 'Real' if prediction == 1 else 'Fake'

    # Explainability (WHY)
    if prediction == 0:
        reason = (
            "The article shows linguistic patterns commonly linked to misinformation, "
            "such as emotional tone, exaggeration, or unverifiable claims."
        )
    else:
        reason = (
            "The article maintains neutral language and keyword patterns "
            "typically found in credible journalism."
        )

    # Keyword influence
    vectorized_text = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    coef = model.coef_[0]

    tfidf_values = vectorized_text.toarray()[0]
    influence = np.abs(tfidf_values * coef)

    top_indices = influence.argsort()[-6:][::-1]
    keywords = [feature_names[i] for i in top_indices if influence[i] > 0]

    return jsonify({
        'label': label,
        'confidence': round(confidence, 2),
        'keywords': keywords,
        'reason': reason
    })


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
