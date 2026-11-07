param(
    [string]$OutputRoot = 'E:\LJQ_code\LJQ_code_experiment_storage\lt_abc_v1\20261106_local_01',
    [string]$Init = 'E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2\seed_2039\backbone\best_model.pth',
    [string]$HistoricalRoot = 'E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1',
    [string]$CondaExecutable = 'conda',
    [string]$CondaEnvironment = 'PY10'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
& $CondaExecutable run --no-capture-output -n $CondaEnvironment python -u `
    (Join-Path $PSScriptRoot 'run_lt_abc.py') --stage pipeline --output $OutputRoot --init $Init `
    --historical-pool (Join-Path $HistoricalRoot 'closedset_9class_train.mat') `
    --historical-pool (Join-Path $HistoricalRoot 'closedset_9class_test.mat') `
    *> (Join-Path $OutputRoot 'pipeline.log')
exit $LASTEXITCODE
