# ============================================
# Network Anomaly Detection For Insider Attacks - Auto Start
# ============================================

Write-Host "=" -ForegroundColor Cyan
Write-Host " Starting Network Anomaly Detection System" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan

# تحديد المسار الحالي للمشروع لضمان عدم حدوث خطأ في الانتقال بين المجلدات
$BASE_DIR = $PSScriptRoot

# 1. Check Modules
if (-not (Test-Path "$BASE_DIR\Back-End\models\random_forest.pkl")) {
    Write-Host "  Models not found! Training first..." -ForegroundColor Yellow
    cd "$BASE_DIR\Back-End"
    .\venv\Scripts\Activate.ps1
    python Train_Models.py
    cd ..
    Write-Host " Training complete!" -ForegroundColor Green
}

# 2. Start Back-End
Write-Host "`n Starting Backend API..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BASE_DIR\Back-End'; .\venv\Scripts\Activate.ps1; python App.py"

# Wait 5 Seconds
Write-Host " Waiting 5 seconds for Backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 3. Start Front-End
Write-Host " Starting Frontend Dashboard..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BASE_DIR\Front-End'; npm start"

Write-Host "`n System Started!" -ForegroundColor Green
Write-Host " Dashboard will open at: http://localhost:3000" -ForegroundColor White
Write-Host " API running at: http://localhost:5000" -ForegroundColor White
Write-Host "`n Keep both PowerShell windows open!" -ForegroundColor Yellow