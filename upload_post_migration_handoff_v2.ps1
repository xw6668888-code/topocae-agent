param(
    [string]$SshTarget = "ywx1@node6",
    [string]$RemoteDir = "/home/ywx1/dataset_generator"
)

$ErrorActionPreference = "Stop"

if ($RemoteDir.Contains("'")) {
    throw "RemoteDir must not contain a single quote."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$checksumName = "POST_MIGRATION_HANDOFF_CHECKSUMS_V2.sha256"
$files = @(
    "POST_MIGRATION_PROJECT_SUPERVISION_AND_HYBRID_CAE_PLAN.zh-CN.md",
    "LINUX_SUPERVISOR_CONTINUATION_PROMPT.zh-CN.md",
    "LINUX_SUPERVISOR_CONTINUATION_PROMPT_V2.zh-CN.md",
    "GOAL_P4_R0_RUNTIME_PREPARATION.zh-CN.md",
    "CODEX_CONTEXT_2026_07_28_POST_MIGRATION_HANDOFF.md",
    "NON_GIT_WORKSPACE_DECLARATION.zh-CN.md",
    "POST_MIGRATION_HANDOFF_DELIVERY_V2.zh-CN.md",
    $checksumName
)

foreach ($commandName in @("ssh", "scp")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$commandName' was not found in PATH."
    }
}

$localPaths = foreach ($file in $files) {
    $path = Join-Path $projectRoot $file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required handoff file is missing: $path"
    }
    $path
}

Write-Host "[1/4] Verifying local handoff checksums..."
$checksumPath = Join-Path $projectRoot $checksumName
$verifiedCount = 0

foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') {
        throw "Invalid checksum line: $line"
    }

    $expectedHash = $Matches[1].ToLowerInvariant()
    $fileName = $Matches[2]
    $filePath = Join-Path $projectRoot $fileName

    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        throw "Checksum target is missing: $filePath"
    }

    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Checksum mismatch: $fileName"
    }

    Write-Host "  OK  $fileName"
    $verifiedCount++
}

if ($verifiedCount -ne 7) {
    throw "Expected 7 checksum entries, but found $verifiedCount."
}

Write-Host "[2/4] Checking remote project directory..."
& ssh $SshTarget "test -d '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Remote directory is unavailable: ${SshTarget}:$RemoteDir"
}

Write-Host "[3/4] Uploading 8 handoff files..."
& scp -- @localPaths "${SshTarget}:$RemoteDir/"
if ($LASTEXITCODE -ne 0) {
    throw "SCP upload failed with exit code $LASTEXITCODE."
}

Write-Host "[4/4] Verifying uploaded files on Linux..."
& ssh $SshTarget "cd '$RemoteDir' && sha256sum -c '$checksumName'"
if ($LASTEXITCODE -ne 0) {
    throw "Remote checksum verification failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "POST-MIGRATION HANDOFF UPLOAD: PASS"
Write-Host "Uploaded files: 8"
Write-Host "Verified content files: 7/7"
Write-Host "Remote root: ${SshTarget}:$RemoteDir"

