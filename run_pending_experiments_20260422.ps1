$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\RELATED_DATA\XGATE_Public"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$canonicalRoot = Join-Path $projectRoot "results\canonical_multirun_fixed_20260420"
$ablationRoot = Join-Path $projectRoot "results\ablation"
$claimClosureRoot = Join-Path $projectRoot "results\claim_closure"

Set-Location $projectRoot

function Write-Stamp {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "$ts | $Message"
}

function Test-CompletionMarker {
    param([string]$Path, [string]$MarkerName)
    return (Test-Path (Join-Path $Path $MarkerName))
}

function Archive-IfPartial {
    param([string]$Path, [string]$MarkerName)

    if (-not (Test-Path $Path)) {
        return
    }

    if (Test-CompletionMarker -Path $Path -MarkerName $MarkerName) {
        Write-Stamp "Found completed output at $Path ($MarkerName present). Skipping reset."
        return
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archivePath = "${Path}_stale_${timestamp}"
    Write-Stamp "Found partial output at $Path without $MarkerName. Archiving to $archivePath."
    Move-Item -LiteralPath $Path -Destination $archivePath
}

Write-Stamp "Queue runner started."

if (-not (Test-CompletionMarker -Path $canonicalRoot -MarkerName "FINAL_STATS.json")) {
    throw "Canonical results are missing FINAL_STATS.json at $canonicalRoot. Refusing to rerun canonical automatically."
}

if (Test-CompletionMarker -Path $ablationRoot -MarkerName "ABLATION_STATS.json") {
    Write-Stamp "Full ablation already completed. Skipping ablation."
} else {
    Archive-IfPartial -Path $ablationRoot -MarkerName "ABLATION_STATS.json"
    Write-Stamp "Launching full ablation study."
    & $python "run_ablation_study.py" `
        --device "cuda" `
        --teacher-root "results/canonical_multirun_fixed_20260420" `
        --results-root "results/ablation"

    $ablationExit = $LASTEXITCODE
    Write-Stamp "Full ablation exited with code $ablationExit."

    if ($ablationExit -ne 0) {
        exit $ablationExit
    }
}

if (Test-CompletionMarker -Path $claimClosureRoot -MarkerName "CLAIM_CLOSURE_STATS.json") {
    Write-Stamp "Claim-closure evaluation already completed. Skipping claim closure."
    exit 0
}

Archive-IfPartial -Path $claimClosureRoot -MarkerName "CLAIM_CLOSURE_STATS.json"
Write-Stamp "Launching claim-closure evaluation."
& $python "run_claim_closure.py" `
    --results-root "results/canonical_multirun_fixed_20260420" `
    --output-root "results/claim_closure"

$claimExit = $LASTEXITCODE
Write-Stamp "Claim closure exited with code $claimExit."
exit $claimExit
