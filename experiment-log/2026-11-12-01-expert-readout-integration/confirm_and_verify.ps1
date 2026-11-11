$ErrorActionPreference = 'Stop'
$experimentDir = $PSScriptRoot
$protocolPath = Join-Path $experimentDir 'protocol.json'
$campaign = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
$outputDir = $campaign.storage
$pythonExe = 'D:\anaconda3\envs\PY10\python.exe'
$matlabExe = 'D:\Matlab\bin\matlab.exe'
Set-Location -LiteralPath $campaign.repository

function Invoke-PythonStep {
    param([string]$StepName, [string]$ScriptName, [string[]]$StepArguments = @())
    $logPath = Join-Path $outputDir ($StepName + '.log')
    & $pythonExe -u (Join-Path $experimentDir $ScriptName) @StepArguments *> $logPath
    if ($LASTEXITCODE -ne 0) { throw "Failed $StepName; inspect $logPath" }
    Write-Output "STEP_COMPLETE $StepName"
}

function Invoke-MatlabStep {
    param([string]$StepName, [string]$Body)
    $escapedDir = $experimentDir.Replace("'", "''")
    $statement = "addpath('$escapedDir'); $Body"
    $logPath = Join-Path $outputDir ($StepName + '.log')
    & $matlabExe -batch $statement *> $logPath
    if ($LASTEXITCODE -ne 0) { throw "Failed $StepName; inspect $logPath" }
    Write-Output "STEP_COMPLETE $StepName"
}

# Invoke only after all desired training workers have exited and the adaptive
# decision has been recorded. The benchmark rejects active training/scoring.
Invoke-PythonStep 'final_matrix_development_analysis' 'analyze_matrix.py'
Invoke-PythonStep 'final_matrix_development_operating_points' 'matrix_operating_points.py'
Invoke-PythonStep 'stored_identity_registry' 'audit_identity_registry.py' @('build')
Invoke-PythonStep 'stored_development_isolation' 'verify_stored_development.py'
Invoke-PythonStep 'supervision_fit_acceptance' 'supervision_control.py' @('fit')
Invoke-PythonStep 'supervision_development_scoring' 'supervision_control.py' @('score')
Invoke-PythonStep 'supervision_development_verification' 'supervision_control.py' @('verify')
Invoke-PythonStep 'supervision_development_runtime' 'supervision_control.py' @('runtime')
Invoke-PythonStep 'budget_audit' 'audit_budget.py'
Invoke-PythonStep 'matrix_contract_verification' 'verify_matrix_contract.py'
Invoke-PythonStep 'near_unity_development_diagnostics' 'diagnose_near_unity.py'
Invoke-PythonStep 'development_background_diagnostics' 'diagnose_backgrounds.py'
foreach ($splitName in @('split1', 'split2')) {
    Invoke-PythonStep ($splitName + '_final_development_runtime') 'runtime.py' @('--split', $splitName)
}
Invoke-PythonStep 'final_runtime_export' 'export_runtime.py'
Invoke-PythonStep 'final_runtime_development_verification' 'export_runtime.py' @('--verify')
Invoke-PythonStep 'runtime_benchmark' 'benchmark_runtime.py'
Invoke-PythonStep 'final_freeze' 'freeze_final.py'

$escapedProtocol = $protocolPath.Replace("'", "''")
Invoke-MatlabStep 'confirmation_generation' "generate_pools('$escapedProtocol','split1',{'confirmation_pool'}); generate_pools('$escapedProtocol','split2',{'confirmation_pool'});"
foreach ($splitName in @('split1', 'split2')) {
    Invoke-PythonStep ($splitName + '_confirmation_acceptance') 'prepare_roles.py' @('--split', $splitName, '--role', 'confirmation')
}
Invoke-PythonStep 'confirmation_stored_identity_verification' 'audit_identity_registry.py' @('confirm')
foreach ($splitName in @('split1', 'split2')) {
    Invoke-PythonStep ($splitName + '_confirmation_scoring') 'factorial_fusion.py' @('score', '--split', $splitName, '--role', 'confirmation')
    Invoke-PythonStep ($splitName + '_confirmation_analysis') 'analyze_factorial.py' @('--split', $splitName, '--role', 'confirmation')
    Invoke-PythonStep ($splitName + '_confirmation_runtime') 'runtime.py' @('--split', $splitName, '--role', 'confirmation')
    Invoke-PythonStep ($splitName + '_matrix_confirmation') 'score_matrix.py' @('--split', $splitName)
}
Invoke-PythonStep 'supervision_confirmation_scoring' 'supervision_control.py' @('score', '--role', 'confirmation')
Invoke-PythonStep 'supervision_confirmation_verification' 'supervision_control.py' @('verify', '--role', 'confirmation')
Invoke-PythonStep 'supervision_confirmation_runtime' 'supervision_control.py' @('runtime', '--role', 'confirmation')
Invoke-PythonStep 'matrix_confirmation_analysis' 'analyze_matrix.py' @('--role', 'confirmation')
Invoke-PythonStep 'matrix_confirmation_operating_points' 'matrix_operating_points.py' @('--role', 'confirmation')
Invoke-MatlabStep 'physical_confirmation_generation' "generate_physical_pairs('$escapedProtocol','split1','confirmation'); generate_physical_pairs('$escapedProtocol','split2','confirmation');"
foreach ($splitName in @('split1', 'split2')) {
    Invoke-PythonStep ($splitName + '_physical_confirmation_scoring') 'evaluate_physical.py' @('--split', $splitName, '--role', 'confirmation')
}
Invoke-PythonStep 'confirmation_core_verification' 'verify_core.py' @('--role', 'confirmation')
Invoke-PythonStep 'confirmation_physical_verification' 'verify_physical.py' @('--role', 'confirmation')
Invoke-PythonStep 'confirmation_readout_diagnostics' 'diagnose_readouts.py' @('--role', 'confirmation')
Invoke-PythonStep 'confirmation_background_diagnostics' 'diagnose_backgrounds.py' @('--role', 'confirmation')
Invoke-PythonStep 'confirmation_portable_verification' 'export_runtime.py' @('--verify', '--role', 'confirmation')
Invoke-PythonStep 'matrix_operating_points_verification' 'verify_operating_points.py'
Invoke-PythonStep 'report_build' 'build_report.py'
Invoke-PythonStep 'figures_build' 'plot_results.py'
Write-Output 'CONFIRMATION_PIPELINE_COMPLETE: inspect results and figures before curation/publication'
