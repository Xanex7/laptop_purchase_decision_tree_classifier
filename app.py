import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAMES = [
    "model_decision_tree_classifier.pkl",
    "model_decision_tree_classifier_2.pkl",
    "model_decision_tree.pkl", 
    "model_dt.pkl"
]

MODEL_PATH = None
for filename in MODEL_FILENAMES:
    file_path = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(file_path):
        MODEL_PATH = file_path
        break

model = None
if MODEL_PATH:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Purchase Predictor | Decision AI</title>
    <!-- Google Fonts, FontAwesome, jsPDF -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>

    <style>
        [data-theme="emerald"] {
            --bg-dark: #02120a;
            --card-glass: rgba(6, 30, 18, 0.75);
            --card-border: rgba(16, 185, 129, 0.25);
            --primary-glow: #10b981;
            --primary-bright: #34d399;
            --primary-dim: rgba(16, 185, 129, 0.15);
        }

        [data-theme="cyberpunk"] {
            --bg-dark: #030712;
            --card-glass: rgba(15, 23, 42, 0.75);
            --card-border: rgba(0, 242, 254, 0.25);
            --primary-glow: #00f2fe;
            --primary-bright: #38bdf8;
            --primary-dim: rgba(0, 242, 254, 0.15);
        }

        [data-theme="amber"] {
            --bg-dark: #0a0d14;
            --card-glass: rgba(18, 24, 38, 0.75);
            --card-border: rgba(245, 158, 11, 0.25);
            --primary-glow: #f59e0b;
            --primary-bright: #fbbf24;
            --primary-dim: rgba(245, 158, 11, 0.15);
        }

        [data-theme="frost"] {
            --bg-dark: #0f172a;
            --card-glass: rgba(30, 41, 59, 0.75);
            --card-border: rgba(129, 140, 248, 0.25);
            --primary-glow: #818cf8;
            --primary-bright: #a5b4fc;
            --primary-dim: rgba(129, 140, 248, 0.15);
        }

        [data-theme="crimson"] {
            --bg-dark: #120307;
            --card-glass: rgba(30, 8, 15, 0.75);
            --card-border: rgba(244, 63, 94, 0.25);
            --primary-glow: #f43f5e;
            --primary-bright: #fb7185;
            --primary-dim: rgba(244, 63, 94, 0.15);
        }

        :root {
            --emerald-glow: #10b981;
            --rose-glow: #f43f5e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(10, 13, 20, 0.65);
            --input-border: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        .ambient-orb {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(140px);
            z-index: -1;
            opacity: 0.25;
            pointer-events: none;
        }
        .orb-1 { top: -200px; right: -100px; background: var(--primary-glow); }
        .orb-2 { bottom: -200px; left: -100px; background: var(--primary-glow); }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 6%;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 13, 20, 0.8);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--primary-glow), var(--primary-bright));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0a0d14;
            font-size: 1.2rem;
            box-shadow: 0 0 20px var(--primary-dim);
        }

        .theme-switcher {
            display: flex;
            gap: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 6px 10px;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .theme-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .theme-btn.active {
            border-color: #ffffff;
            transform: scale(1.2);
        }

        .theme-matrix { background: #10b981; }
        .theme-cyber { background: #00f2fe; }
        .theme-amber { background: #f59e0b; }
        .theme-frost { background: #818cf8; }
        .theme-crimson { background: #f43f5e; }

        .workspace-container {
            max-width: 1400px;
            margin: 35px auto;
            padding: 0 4%;
            width: 100%;
            flex-grow: 1;
        }

        .header-section {
            text-align: center;
            margin-bottom: 38px;
        }

        .title-badge {
            display: inline-block;
            padding: 6px 18px;
            background: var(--primary-dim);
            border: 1px solid var(--primary-glow);
            border-radius: 30px;
            color: var(--primary-bright);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #ffffff 0%, var(--primary-bright) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .main-subtitle {
            color: var(--text-sub);
            font-size: 1.02rem;
            max-width: 650px;
            margin: 0 auto;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 32px;
            align-items: start;
        }

        .sticky-sidebar {
            position: sticky;
            top: 110px;
            z-index: 10;
        }

        .glass-card {
            background: var(--card-glass);
            border-radius: 28px;
            border: 1px solid var(--card-border);
            padding: 34px;
            backdrop-filter: blur(24px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
            margin-bottom: 24px;
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title i {
            color: var(--primary-glow);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .live-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--primary-bright);
            font-weight: 700;
        }

        input, select {
            width: 100%;
            padding: 13px 15px;
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.92rem;
            font-weight: 500;
            transition: all 0.25s ease;
        }

        select {
            appearance: none;
            cursor: pointer;
            padding-right: 36px;
        }

        .select-wrapper {
            position: relative;
        }

        .select-wrapper::after {
            content: '\f107';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            right: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-sub);
            pointer-events: none;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-glow);
            box-shadow: 0 0 0 4px var(--primary-dim);
            background-color: rgba(10, 13, 20, 0.9);
        }

        .btn-predict {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, var(--primary-glow) 0%, var(--primary-bright) 100%);
            color: #0a0d14;
            font-weight: 800;
            font-size: 1.05rem;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            box-shadow: 0 10px 30px var(--primary-dim);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px var(--primary-dim);
            color: #ffffff;
        }

        .valuation-card {
            background: linear-gradient(180deg, var(--primary-dim) 0%, rgba(10, 13, 20, 0.05) 100%);
            border: 1px solid var(--primary-glow);
            border-radius: 20px;
            padding: 24px;
            text-align: center;
        }

        .val-tag {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-sub);
            font-weight: 700;
            margin-bottom: 6px;
        }

        .val-price {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .progress-container {
            margin-top: 16px;
            text-align: left;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-sub);
            margin-bottom: 6px;
        }

        .progress-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--primary-glow);
            width: 0%;
            transition: width 0.4s ease;
        }

        .history-box {
            margin-top: 20px;
            max-height: 220px;
            overflow-y: auto;
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            text-align: left;
        }

        .history-table th {
            color: var(--text-sub);
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.7rem;
        }

        .history-table td {
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
        }

        .tag-yes { color: var(--emerald-glow); font-weight: 700; }
        .tag-no { color: var(--rose-glow); font-weight: 700; }

        .btn-report {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.88rem;
            cursor: pointer;
            margin-top: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.25s ease;
        }

        .btn-report:hover {
            background: var(--primary-dim);
            border-color: var(--primary-glow);
            color: var(--primary-bright);
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(10, 13, 20, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        footer {
            text-align: center;
            padding: 24px;
            color: var(--text-sub);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            margin-top: auto;
        }

        @media (max-width: 1024px) {
            .grid-layout { grid-template-columns: 1fr; }
            .form-grid { grid-template-columns: 1fr; }
            .sticky-sidebar { position: static; }
        }
    </style>
</head>
<body>

<div class="ambient-orb orb-1"></div>
<div class="ambient-orb orb-2"></div>

<nav class="navbar">
    <div class="brand">
        <div class="brand-logo"><i class="fa-solid fa-laptop"></i></div>
        <span>LaptopPurchase<span style="color: var(--primary-glow);">.ai</span></span>
    </div>
    
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="theme-switcher">
            <button class="theme-btn theme-matrix active" onclick="switchTheme('emerald')"></button>
            <button class="theme-btn theme-cyber" onclick="switchTheme('cyberpunk')"></button>
            <button class="theme-btn theme-amber" onclick="switchTheme('amber')"></button>
            <button class="theme-btn theme-frost" onclick="switchTheme('frost')"></button>
            <button class="theme-btn theme-crimson" onclick="switchTheme('crimson')"></button>
        </div>
    </div>
</nav>

<div class="workspace-container">
    <div class="header-section">
        <span class="title-badge">Decision Tree Intelligence</span>
        <h1 class="main-title">Laptop Purchase Predictor</h1>
        <p class="main-subtitle">Synthesize demographic and socioeconomic variables into real-time purchase decision paths.</p>
    </div>

    <div class="grid-layout">
        <!-- Input Form -->
        <div class="glass-card">
            <div class="section-header">
                <span class="section-title"><i class="fa-solid fa-sliders"></i> Classification Attributes</span>
                <span style="font-size: 0.8rem; color: var(--text-sub);">Laptop Users Dataset</span>
            </div>
            
            <form id="predictionForm">
                <div class="form-grid">
                    
                    <div class="form-group">
                        <label>Age <span class="live-val" id="ageVal">30</span></label>
                        <input type="number" name="Age" value="30" min="10" max="100" required oninput="document.getElementById('ageVal').textContent = this.value">
                    </div>

                    <div class="form-group">
                        <label>Gender Orientation</label>
                        <div class="select-wrapper">
                            <select name="Gender" required>
                                <option value="male" selected>Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Region Hub</label>
                        <div class="select-wrapper">
                            <select name="Region" required>
                                <option value="city" selected>City</option>
                                <option value="countryside">Countryside</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Occupation Sector</label>
                        <div class="select-wrapper">
                            <select name="Occupation" required>
                                <option value="student">Student</option>
                                <option value="teacher" selected>Teacher</option>
                                <option value="banker">Banker</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group" style="grid-column: span 2;">
                        <label>Annual Income ($) <span class="live-val" id="incomeVal">$25,000</span></label>
                        <input type="number" name="Income" value="25000" step="500" min="0" required oninput="document.getElementById('incomeVal').textContent = '$' + Number(this.value).toLocaleString()">
                    </div>

                </div>

                <button type="submit" class="btn-predict" id="submitBtn">
                    <span class="btn-text">Execute Decision Path</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </form>
        </div>

        <!-- Telemetry Sidebar -->
        <div class="sticky-sidebar" id="outputSection">
            <div class="glass-card">
                <div class="section-header">
                    <span class="section-title"><i class="fa-solid fa-chart-pie"></i> Prediction Telemetry</span>
                </div>
                
                <div class="valuation-card" id="resultCard">
                    <div class="val-tag">Has Laptop Forecast</div>
                    <div class="val-price">
                        <i class="fa-solid fa-laptop-code" id="resultIcon" style="color: var(--primary-glow);"></i>
                        <span id="resultOutput">Awaiting Input</span>
                    </div>
                    
                    <div class="progress-container">
                        <div class="progress-label">
                            <span>Leaf Node Confidence</span>
                            <strong id="probLabel">0%</strong>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" id="probFill"></div>
                        </div>
                    </div>
                </div>

                <div style="margin-top: 24px;">
                    <div class="section-title" style="font-size: 1rem;"><i class="fa-solid fa-clock-rotate-left"></i> Decision Audit Log</div>
                    <div class="history-box">
                        <table class="history-table">
                            <thead>
                                <tr><th>Time</th><th>Age</th><th>Income</th><th>Has Laptop</th></tr>
                            </thead>
                            <tbody id="historyLog">
                                <tr><td colspan="4" style="color: var(--text-sub); text-align: center;">No predictions logged.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <button type="button" class="btn-report" onclick="downloadPDFReport()">
                    <i class="fa-solid fa-file-pdf"></i> Download Decision Brief
                </button>
            </div>
        </div>

    </div>
</div>

<footer>&copy; 2026 Laptop Purchase Predictor &bull; DecisionTree Core Engine</footer>

<script>
    let lastResult = "Awaiting Input";
    let historyRecords = [];

    function switchTheme(themeName) {
        document.documentElement.setAttribute('data-theme', themeName);
        document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
        
        let themeMap = { 'emerald': 'matrix', 'cyberpunk': 'cyber', 'amber': 'amber', 'frost': 'frost', 'crimson': 'crimson' };
        document.querySelector(`.theme-${themeMap[themeName]}`).classList.add('active');
    }

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const form = e.target;
        const submitBtn = document.getElementById('submitBtn');
        const spinner = document.getElementById('btnSpinner');
        const btnText = submitBtn.querySelector('.btn-text');
        const resultOutput = document.getElementById('resultOutput');
        const resultIcon = document.getElementById('resultIcon');
        const resultCard = document.getElementById('resultCard');
        const probLabel = document.getElementById('probLabel');
        const probFill = document.getElementById('probFill');
        
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        btnText.textContent = 'Traversing Decision Tree...';
        
        try {
            const response = await fetch('/predict', { method: 'POST', body: new FormData(form) });
            const data = await response.json();
            
            if (data.status === 'success') {
                lastResult = data.prediction;
                resultOutput.textContent = data.prediction;
                let probPct = (data.probability * 100).toFixed(1) + '%';
                probLabel.textContent = probPct;
                probFill.style.width = probPct;
                
                if (data.prediction === "Yes") {
                    resultCard.style.borderColor = "rgba(16, 185, 129, 0.4)";
                    resultIcon.style.color = "var(--emerald-glow)";
                    probFill.style.background = "var(--emerald-glow)";
                } else {
                    resultCard.style.borderColor = "rgba(244, 63, 94, 0.4)";
                    resultIcon.style.color = "var(--rose-glow)";
                    probFill.style.background = "var(--rose-glow)";
                }
                
                addHistoryRecord(
                    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), 
                    form.elements['Age'].value, 
                    '$' + Number(form.elements['Income'].value).toLocaleString(), 
                    data.prediction
                );
            } else {
                resultOutput.textContent = 'Error: ' + data.message;
            }
        } catch (error) {
            resultOutput.textContent = error.message;
        } finally {
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Execute Decision Path';
        }
    });

    function addHistoryRecord(time, age, income, result) {
        historyRecords.unshift({ time, age, income, result });
        const historyLog = document.getElementById('historyLog');
        historyLog.innerHTML = '';
        historyRecords.forEach(rec => {
            const tr = document.createElement('tr');
            let isYes = rec.result === 'Yes';
            tr.innerHTML = `<td>${rec.time}</td><td>${rec.age}</td><td>${rec.income}</td><td class="${isYes ? 'tag-yes' : 'tag-no'}">${rec.result}</td>`;
            historyLog.appendChild(tr);
        });
    }

    function downloadPDFReport() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        doc.setFont("helvetica", "bold");
        doc.setFontSize(20);
        doc.text("Laptop Purchase Decision Brief", 20, 22);
        doc.setFontSize(10);
        doc.text("Date: " + new Date().toLocaleDateString(), 20, 30);
        doc.line(20, 35, 190, 35);
        doc.setFontSize(14);
        doc.text("Has Laptop Classification: " + lastResult, 20, 48);
        doc.save("Laptop_Purchase_Report.pdf");
    }
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Decision Tree model not loaded.'}), 500
        
    try:
        # Encodings matching model split rules
        gender_map = {'male': 1, 'female': 0}
        region_map = {'city': 0, 'countryside': 1}
        occupation_map = {'student': 0, 'teacher': 1, 'banker': 2}
        
        gender_raw = request.form['Gender'].lower()
        region_raw = request.form['Region'].lower()
        occupation_raw = request.form['Occupation'].lower()
        
        gender_val = gender_map.get(gender_raw, 1)
        region_val = region_map.get(region_raw, 0)
        occupation_val = occupation_map.get(occupation_raw, 0)

        data_dict = {
            'Age': [float(request.form['Age'])],
            'Gender': [gender_val],
            'Region': [region_val],
            'Occupation': [occupation_val],
            'Income': [float(request.form['Income'])]
        }
        
        features_df = pd.DataFrame(data_dict)
        raw_pred = int(model.predict(features_df)[0])
        
        prob_score = 0.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            prob_score = float(probs[raw_pred])
            
        label_mapping = { 
            1: "Yes", 
            0: "No" 
        }
        final_output = label_mapping.get(raw_pred, "Yes" if raw_pred == 1 else "No")
        
        return jsonify({
            'status': 'success',
            'prediction': final_output,
            'probability': round(prob_score, 3)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
