param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$SourceDatabaseUrl = $env:SOURCE_DATABASE_URL,
    [string]$TargetDatabaseUrl = $env:TARGET_DATABASE_URL,
    [string]$BackendRoot = "",
    [string]$SnapshotDir = "",
    [switch]$SkipImport,
    [switch]$IncludeEncryptedWorkspaceSecrets,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BackendRoot)) {
    $BackendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$pythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$skillScript = "C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\scripts\copy_project_snapshot.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "No existe $pythonExe. Prepara primero el virtualenv del backend."
}

if (-not (Test-Path -LiteralPath $skillScript)) {
    throw "No se encontro el script del skill en $skillScript."
}

if ([string]::IsNullOrWhiteSpace($SourceDatabaseUrl)) {
    throw "Debes pasar -SourceDatabaseUrl o definir SOURCE_DATABASE_URL en la sesion."
}

$resolvedSnapshotDir = if ([string]::IsNullOrWhiteSpace($SnapshotDir)) {
    Join-Path $BackendRoot ("runtime\prod-session-{0}" -f $ProjectId)
} else {
    $SnapshotDir
}

$descriptorJson = & $pythonExe -c "import json, sys; from sqlalchemy.engine import make_url; parsed = make_url(sys.argv[1]); print(json.dumps({'drivername': parsed.drivername, 'host': parsed.host or '', 'port': parsed.port or '', 'database': parsed.database or ''}))" $SourceDatabaseUrl
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo interpretar -SourceDatabaseUrl."
}

$descriptor = $descriptorJson | ConvertFrom-Json

Write-Host "Backend root: $BackendRoot"
Write-Host "Snapshot dir: $resolvedSnapshotDir"
Write-Host "Source driver: $($descriptor.drivername)"
Write-Host "Source host: $($descriptor.host)"
Write-Host "Source port: $($descriptor.port)"
Write-Host "Source database: $($descriptor.database)"
Write-Host "Import to local: $([bool](-not $SkipImport))"
Write-Host "Include encrypted workspace secrets: $([bool]$IncludeEncryptedWorkspaceSecrets)"

if ($DryRun) {
    Write-Host "Ejecucion simulada. No se exporto ni importo informacion."
    exit 0
}

$commandArgs = @(
    $skillScript,
    "--source-database-url",
    $SourceDatabaseUrl,
    "--project-id",
    $ProjectId,
    "--backend-root",
    $BackendRoot,
    "--snapshot-dir",
    $resolvedSnapshotDir
)

if (-not [string]::IsNullOrWhiteSpace($TargetDatabaseUrl)) {
    $commandArgs += @("--target-database-url", $TargetDatabaseUrl)
}
if ($SkipImport) {
    $commandArgs += "--skip-import"
}
if ($IncludeEncryptedWorkspaceSecrets) {
    $commandArgs += "--include-encrypted-workspace-secrets"
}

& $pythonExe @commandArgs
exit $LASTEXITCODE
