# BEVFormerRadar — density_fix 전체 실험 (Exp A / B / C)
# 실행: powershell -ExecutionPolicy Bypass -File run_density_fix.ps1

$ErrorActionPreference = "Continue"
$SYNTH_DIR  = "$PSScriptRoot\..\NeuralRadarSim\outputs\synthetic\scene_00"
$CKPT_ROOT  = "checkpoints/density_fix"
$LOG_DIR    = "logs"

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

# Exp A — Real only baseline
Run-Exp "density_fix_exp_a" @("--ratio", "0", "--ckpt-root", $CKPT_ROOT)

# Exp B — Real + Synth 1:1
Run-Exp "density_fix_exp_b" @("--synth-dir", $SYNTH_DIR, "--ratio", "1", "--ckpt-root", $CKPT_ROOT)

# Exp C — Real + Synth 1:3
Run-Exp "density_fix_exp_c" @("--synth-dir", $SYNTH_DIR, "--ratio", "3", "--ckpt-root", $CKPT_ROOT)

# 평가
Write-Host ""
Write-Host "============================================================"
Write-Host "  Running evaluate_all.py"
Write-Host "============================================================"
python evaluate_all.py --ckpt-root $CKPT_ROOT 2>&1 | Tee-Object -FilePath "$LOG_DIR\density_fix_eval.log"

Write-Host ""
Write-Host "All done. Results in logs/ and results/"
