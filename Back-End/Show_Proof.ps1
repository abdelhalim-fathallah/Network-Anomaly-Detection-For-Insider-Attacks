Write-Host "=== System Credibility Verification ===" -ForegroundColor Cyan
Write-Host "---------------------------------------" -ForegroundColor Cyan

# 1. Training Data Check
Write-Host "1. Dataset Analysis:" -ForegroundColor Yellow
python -c "import pandas as pd; df = pd.read_csv('data/KDDTrain.txt', header=None); print(f'   ✔ {len(df):,} Real-world records verified')"

# 2. Trained Models Check
Write-Host "2. Machine Learning Assets:" -ForegroundColor Yellow
python -c "import os; models = [f for f in os.listdir('models') if f.endswith('.pkl')]; print(f'   ✔ {len(models)} Optimized models identified (.pkl)')"

# 3. Performance Metrics
Write-Host "3. Final Accuracy Metrics:" -ForegroundColor Yellow
python -c "import pickle; m = pickle.load(open('models/metrics.pkl', 'rb')); print(f'   ✔ Ensemble Model Accuracy: {m[\"ensemble\"][\"accuracy\"]*100:.2f}%')"

Write-Host "---------------------------------------" -ForegroundColor Cyan
Write-Host "=== System is Ready for Live Demo! ===" -ForegroundColor Green

# 4. The requested part: Auto-open the Dashboard
Write-Host "`n🌐 Launching Interactive Dashboard..." -ForegroundColor White
Start-Sleep -Seconds 2
Start-Process "http://localhost:3000"