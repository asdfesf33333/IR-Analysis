import os, pandas as pd, numpy as np, calendar
from datetime import date
from flask import Flask, request, jsonify, render_template_string
from factor_analyzer import FactorAnalyzer

# --- CONFIG ---
PORT = 5000
YEAR = 2026
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- THE DASHBOARD (HTML/CSS/JS) ---
INDEX_HTML = """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
<style>
    :root { --bg: #0f172a; --accent: #818cf8; --text: #f1f5f9; }
    body { background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif; padding: 20px; }
    .card { background: rgba(30,41,59,0.7); border: 1px solid #334155; border-radius: 15px; padding: 20px; margin-bottom: 20px; }
    .btn { background: var(--accent); color: white; border: none; padding: 12px; border-radius: 8px; width: 100%; cursor: pointer; font-weight: bold; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th { text-align: left; color: var(--accent); padding: 10px; border-bottom: 1px solid #334155; }
    td { padding: 10px; border-bottom: 1px solid #334155; font-size: 0.9rem; }
</style></head><body>
    <h1 style="text-align:center; margin-bottom: 30px;">Absenteeism Analytics</h1>
    <div style="display:grid; grid-template-columns: 280px 1fr; gap: 20px;">
        <div><div class="card">
            <input type="file" id="fileInput" style="margin-bottom:10px; width:100%;">
            <button class="btn" onclick="run()">Process Analytics</button>
        </div></div>
        <div>
            <div class="card"><h3>Weekly Patterns</h3><div id="plot" style="height:400px;"></div></div>
            <div class="card" id="fBox" style="display:none;"><h3>Factor Analysis Loading Table</h3><table id="fTable"></table></div>
        </div>
    </div>
    <script>
        async function run() {
            const input = document.getElementById('fileInput');
            if(!input.files[0]) return alert('Select file');
            const fd = new FormData(); fd.append('file', input.files[0]);
            const res = await fetch('/upload', { method:'POST', body:fd });
            const data = await res.json();
            if(data.error) return alert(data.error);
            
            const days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
            const counts = days.map(d => data.scatter_data.filter(x => x.Weekday === d).length);
            Plotly.newPlot('plot', [{ x:days, y:counts, type:'scatter', mode:'lines+markers', line:{shape:'spline', color:'#818cf8', width:4}}], 
                { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#94a3b8'} });
            
            if(data.factor_table.length > 0) {
                document.getElementById('fBox').style.display='block';
                const keys = Object.keys(data.factor_table[0]);
                let head = '<tr>' + keys.map(k => `<th>${k==='index'?'indicator':k}</th>`).join('') + '</tr>';
                let body = data.factor_table.map(r => '<tr>' + keys.map(k => `<td>${r[k]}</td>`).join('') + '</tr>').join('');
                document.getElementById('fTable').innerHTML = head + body;
            }
        }
    </script>
</body></html>
"""

# --- SERVER CODE ---
app = Flask(__name__)

@app.route('/')
def home(): return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        f = request.files['file']
        p = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(p)
        sheets = pd.read_excel(p, sheet_name=None)
        scatter = []
        matrix = []
        for name, df in sheets.items():
            df.columns = [str(c).strip() for c in df.columns]
            dates = [c for c in df.columns if c.isdigit()]
            for _, row in df.iterrows():
                abs_row = []
                for d in dates:
                    is_p = (str(row[d]).strip().upper() == 'P-P')
                    try:
                        m_num = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}.get(name.lower())
                        wk = date(YEAR, m_num, int(d)).strftime('%A')
                        if not is_p: scatter.append({'Weekday': wk})
                        abs_row.append({'k': f"{name}_{d}", 'v': 0 if is_p else 1})
                    except: pass
                matrix.append(abs_row)
        
        all_k = sorted(list(set(x['k'] for r in matrix for x in r)))
        rows = [[{x['k']: x['v'] for x in r}.get(k,0) for k in all_k] for r in matrix]
        fa_df = pd.DataFrame(rows, columns=all_k)
        fa_df = fa_df.loc[:, (fa_df != fa_df.iloc[0]).any()]
        
        f_table = []
        if not fa_df.empty and fa_df.shape[0] > fa_df.shape[1]:
            fa = FactorAnalyzer(rotation="varimax")
            fa.fit(fa_df)
            n = sum(fa.get_eigenvalues()[0] > 1)
            if n > 0:
                fa = FactorAnalyzer(n_factors=n, rotation="varimax")
                fa.fit(fa_df)
                res = pd.DataFrame(fa.loadings_, columns=[f"F{i+1}" for i in range(n)], index=fa_df.columns)
                f_table = res.reset_index().round(3).to_dict(orient='records')
        return jsonify({'scatter_data': scatter, 'factor_table': f_table})
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=PORT)
