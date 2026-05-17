# Exp A / B / C 재실험 (synth 경로 절대경로 버전)
# 실행: powershell -ExecutionPolicy Bypass -File run_bc.ps1

$ErrorActionPreference = "Continue"
$SYNTH_DIR = "$PSScriptRoot\..\NeuralSensorSim\outputs\synthetic\scene_00"
$LOG_DIR   = "logs"

Set-Location $PSScriptRoot

if (-not (Test-Path $SYNTH_DIR)) {
    Write-Host "[ERROR] Synth dir not found: $SYNTH_DIR"
    exit 1
}
Write-Host "[OK] Synth dir: $SYNTH_DIR"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Run-Exp($Label, $CmdArgs) {
    $log   = "$LOG_DIR\$Label.log"
    $start = Get-Date
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  $Label"
    Write-Host "============================================================"
    & python train_mixed.py @CmdArgs 2>&1 | Tee-Object -FilePath $log
    $elapsed = (Get-Date) - $start
    Write-Host "  [$Label] Done in $($elapsed.ToString('hh\:mm\:ss'))"
}

# Exp A
Run-Exp "exp_a" @("--ratio", "0")

# Exp B
Run-Exp "exp_b_r1" @("--synth-dir", $SYNTH_DIR, "--ratio", "1")

# Exp C
Run-Exp "exp_c_r3" @("--synth-dir", $SYNTH_DIR, "--ratio", "3")

# 평가
Write-Host ""
Write-Host "============================================================"
Write-Host "  evaluate_all.py"
Write-Host "============================================================"
& python evaluate_all.py 2>&1 | Tee-Object -FilePath "$LOG_DIR\evaluate_all.log"

Write-Host "All done."
