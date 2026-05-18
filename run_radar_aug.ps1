# BEVFormerRadar — Exp D: Real Camera + Radar Augmentation
# 실행: powershell -ExecutionPolicy Bypass -File run_radar_aug.ps1

$ErrorActionPreference = "Continue"
$RADAR_AUG_DIR = "C:\Users\Administrator\Documents\project\personal\SensorFusion\NeuralSensorSim\outputs\radar_aug"
$CKPT_ROOT     = "checkpoints/radar_aug"
$LOG_DIR       = "logs"

Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Run-Exp {
    param($Label, $RunArgs)
    $log = "$LOG_DIR\$Label.log"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Starting $Label"
    Write-Host "  Log: $log"
    Write-Host "============================================================"
    $start = Get-Date
    python train_mixed.py @RunArgs 2>&1 | Tee-Object -FilePath $log
    $elapsed = (Get-Date) - $start
    Write-Host "  [$Label] Done in $($elapsed.ToString('hh\:mm\:ss'))"
}

# Step 0: 합성 데이터 생성 (augment_radar_real.py)
Write-Host ""
Write-Host "============================================================"
Write-Host "  Step 0: Generating radar_aug samples..."
Write-Host "============================================================"
$aug_script = "..\NeuralSensorSim\scripts\augment_radar_real.py"
if (Test-Path $aug_script) {
    Push-Location "..\NeuralSensorSim"
    python scripts\augment_radar_real.py --n-donors 2 --seed 42 2>&1 | Tee-Object -FilePath "$PSScriptRoot\$LOG_DIR\radar_aug_gen.log"
    Pop-Location
} else {
    Write-Host "  [SKIP] augment_radar_real.py not found at $aug_script"
    Write-Host "  Assuming $RADAR_AUG_DIR already exists."
}

# Exp D-1: Real only baseline (재확인)
Run-Exp "radar_aug_exp_a" @("--ratio", "0", "--ckpt-root", $CKPT_ROOT)

# Exp D-2: Real camera + radar aug 1:1
Run-Exp "radar_aug_exp_d1" @("--radar-aug-dir", $RADAR_AUG_DIR, "--ratio", "1", "--ckpt-root", $CKPT_ROOT)

# Exp D-3: Real camera + radar aug 1:3
Run-Exp "radar_aug_exp_d3" @("--radar-aug-dir", $RADAR_AUG_DIR, "--ratio", "3", "--ckpt-root", $CKPT_ROOT)

# 평가
Write-Host ""
Write-Host "============================================================"
Write-Host "  Running evaluate_all.py"
Write-Host "============================================================"
python evaluate_all.py --ckpt-root $CKPT_ROOT 2>&1 | Tee-Object -FilePath "$LOG_DIR\radar_aug_eval.log"

Write-Host ""
Write-Host "All done. Results in logs/ and results/"
