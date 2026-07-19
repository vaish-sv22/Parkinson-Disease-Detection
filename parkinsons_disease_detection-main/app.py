
from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
from functools import wraps

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"  # change for production

# Load model + scaler
model = joblib.load("parkinsons_voting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define same features used during training
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR'
]

# ---------------------------
# Simple login_required decorator
# ---------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect to prediction
    if session.get("logged_in"):
        return redirect(url_for("prediction_page"))

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # Simple credential check (demo only)
        if username == "admin" and password == "admin":
            session['logged_in'] = True
            flash("Logged in successfully.", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for("prediction_page"))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route('/prediction')
@login_required
def prediction_page():
    return render_template('index.html', features=features)

@app.route('/performance')
@login_required
def performance():
    graph_image = "analysis_graph.png"
    return render_template("performance.html", graph_image=graph_image)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    values = [float(request.form.get(f)) for f in features]
    scaled_values = scaler.transform([values])
    prediction = model.predict(scaled_values)[0]

    graph_image = "graph_graph.png"

    if prediction == 1:
        status = "Parkinson's Detected"
        description = "Based on your input, there are signs that may indicate Parkinson's disease."
        symptoms = [
            "Tremors in hands, arms, legs, or face",
            "Muscle stiffness or rigidity",
            "Slow movements (bradykinesia)",
            "Balance or coordination issues",
            "Changes in speech or handwriting"
        ]
        precautions = [
            "Regular exercise (yoga, stretching, walking)",
            "Healthy diet with fruits, vegetables, and omega-3",
            "Hydration + proper sleep",
            "Take medications regularly",
            "Engage in mental activities"
        ]
        color = "danger"
    else:
        status = "Healthy"
        description = "Your parameters appear within a healthy range."
        symptoms = ["No major symptoms detected"]
        precautions = [
            "Maintain exercise + diet",
            "Good sleep",
            "Avoid smoking & alcohol",
            "Regular health checkups"
        ]
        color = "success"

    return render_template(
        "result.html",
        status=status,
        description=description,
        symptoms=symptoms,
        precautions=precautions,
        color=color,
        graph_image=graph_image
    )

if __name__ == '__main__':
    app.run(debug=True)
