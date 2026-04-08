# 🔥 Wildfire Prediction Portal

An AI-powered web application for real-time wildfire risk prediction, 24-hour forecasting, and email alerting — built with Flask, scikit-learn, XGBoost, and MapLibre GL.

---

## Features

- **Interactive map** — click any location to run a prediction
- **Fire risk classification** — ensemble VotingClassifier (XGBoost + Random Forest + Logistic Regression, 3× balancing strategies)
- **Fire Radiative Power (FRP) estimation** — XGBoost regression model
- **Live weather** — pulled from Open-Meteo at prediction time
- **24-hour forecast** — runs both models on each upcoming hour
- **Explainable AI** — horizontal bar charts showing which features drove the prediction
- **Email alerts** — optional, per-user, with configurable risk threshold
- **Prediction history** — table + trend chart of all past predictions
- **OTP-verified signup** — 6-digit code sent to email, expires in 5 minutes
- **Rate limiting** — per-route limits on predict, forecast, explain, and OTP endpoints

---

## Project Structure

```
Project_Wildfire/
│
├── app.py                          # Flask application — all routes and logic
│
├── .env                            # Your secret credentials (never commit this)
├── .env.example                    # Template — copy to .env and fill in values
├── .gitignore                      # Should include .env and users.db
│
├── voting_ensemble_fire.joblib     # Trained VotingClassifier (classification)
├── fire_regression_model_2.pkl     # Trained XGBRegressor + scaler + feature list
│
├── users.db                        # SQLite database (auto-created on first run)
│
├── templates/                      # Jinja2 HTML templates — served by Flask
│   ├── index.html                  # Landing page
│   ├── login.html                  # Login page
│   ├── signup.html                 # Signup page with OTP
│   ├── dashboard.html              # Main prediction dashboard
│   ├── history.html                # Prediction history + trend chart
│   └── alerts.html                 # Email alert settings
│
├── static/
│   └── images/                     # SVG icons used in navbars and buttons
│       ├── undraw_login_weas_2.svg
│       ├── undraw_sign-here_lxua.svg
│       ├── undraw_season-change_ohe6.svg
│       ├── undraw_log-out_2vod_2.svg
│       ├── undraw_log-out_2vod.svg
│       └── undraw_dashboard_p93p.svg
│
└── requirements.txt                # Python dependencies
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/wildfire-portal.git
cd wildfire-portal
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
SECRET_KEY=your-long-random-secret-key
GMAIL_USER=wildfirepredictor@gmail.com
GMAIL_APP_PASSWORD=your-16-char-app-password
DB_PATH=users.db
FLASK_DEBUG=false
```

To generate a secure `SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

To get a Gmail App Password:
1. Go to [myaccount.google.com](https://myaccount.google.com) → Security
2. Enable 2-Step Verification if not already on
3. Search for "App Passwords" → generate one for Mail
4. Paste the 16-character password into `.env`

### 5. Place your model files

Put these two files in the project root (same folder as `app.py`):

```
voting_ensemble_fire.joblib
fire_regression_model_2.pkl
```

### 6. Run the app

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

> ⚠️ Always open the app via `localhost` — never open `.html` files directly in the browser, as Jinja2 templates will not render correctly.

---

## Dependencies

Install all at once with:

```bash
pip install -r requirements.txt
```

**Core:**
- `flask` — web framework
- `flask-cors` — cross-origin support
- `flask-limiter` — rate limiting
- `python-dotenv` — loads `.env` file
- `werkzeug` — password hashing (included with Flask)

**ML:**
- `joblib` — model loading
- `scikit-learn==1.6.1` — VotingClassifier, RandomForest, LogisticRegression, StandardScaler
- `xgboost` — XGBClassifier and XGBRegressor

**Data & API:**
- `numpy` — array operations
- `requests` — Open-Meteo weather API calls

> ⚠️ The models were trained on scikit-learn `1.6.1`. Using a newer version (like `1.7.x`) will show `InconsistentVersionWarning` on load. Everything will still work, but ask your teammate to retrain and re-export the models when possible.

---

## Environment Variables Reference

| Variable | Description | Example |
|---|---|---|
| `SECRET_KEY` | Flask session encryption key | `abc123...` (32+ chars) |
| `GMAIL_USER` | Gmail address used to send OTPs and alerts | `you@gmail.com` |
| `GMAIL_APP_PASSWORD` | 16-char Gmail App Password | `xxxx xxxx xxxx xxxx` |
| `DB_PATH` | Path to the SQLite database file | `users.db` |
| `FLASK_DEBUG` | Enable Flask debug mode (`true`/`false`) | `false` |

---

## API Routes

| Method | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Landing page |
| `GET/POST` | `/signup` | No | Signup with OTP verification |
| `GET/POST` | `/login` | No | Login |
| `GET` | `/logout` | No | Logout and clear session |
| `POST` | `/generate_otp` | No | Send OTP to email (rate: 5/min) |
| `GET` | `/dashboard` | ✅ | Main dashboard |
| `GET` | `/history` | ✅ | History page |
| `GET` | `/get_history` | ✅ | JSON — past predictions |
| `GET` | `/alerts` | ✅ | Alert settings page |
| `GET` | `/get_alert_settings` | ✅ | JSON — current alert prefs |
| `POST` | `/save_alert_settings` | ✅ | Save alert prefs |
| `POST` | `/predict` | ✅ | Run prediction (rate: 30/min) |
| `POST` | `/forecast` | ✅ | 24-hour forecast (rate: 20/min) |
| `POST` | `/explain` | ✅ | Feature importances (rate: 30/min) |

---

## Notes

- **No data is ever sent to a third party** other than Open-Meteo (weather) and MapTiler (map tiles). Both are read-only API calls with no user data attached.
- **The OTP is never sent to the browser.** It is generated server-side, stored in the session, and emailed directly.
- **Alert emails are only sent when:** alerts are enabled by the user, and the prediction risk meets or exceeds their chosen threshold.
- **The SQLite database** (`users.db`) is auto-created on first run. For production, consider migrating to PostgreSQL.

---

## Deployment (Render.com — Free Tier)

1. Push the project to a GitHub repo (make sure `.env` and `users.db` are in `.gitignore`)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
5. Add all your `.env` variables under **Environment → Add Environment Variable**
6. Add `gunicorn` to `requirements.txt`
7. Deploy

> The free tier spins down after 15 minutes of inactivity — first load after that takes ~30 seconds.

---

## Team

| Role | Contributor |
|---|---|
| Frontend, Backend, APIs, Version Control | Yuvraj |
| Data Collection, Cleaning, Model Training | [Teammate name] |

---

## License

MIT