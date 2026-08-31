param(
    [string]$EnvFile = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $backendRoot ".venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $backendRoot ".env.supabase.local"
}

function Set-ProcessEnvironmentFromFile {
    param([string]$Path)

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $separatorIndex = $line.IndexOf("=")
        if ($separatorIndex -lt 1) {
            continue
        }
        $key = $line.Substring(0, $separatorIndex).Trim()
        $value = $line.Substring($separatorIndex + 1).Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "No existe el archivo de entorno $EnvFile. Crea backend/.env.supabase.local a partir de backend/.env.supabase.local.example."
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "No existe $pythonExe. Prepara primero el virtualenv del backend."
}

Set-ProcessEnvironmentFromFile -Path $EnvFile

$forcedSettings = @{
    "SCHEMA_MANAGEMENT_MODE" = "alembic"
    "RUNTIME_BOOTSTRAP_ENABLED" = "false"
    "KNOWLEDGE_REPO_AUTOSYNC_ENABLED" = "false"
}

$safeDefaults = @{
    "DATABASE_POOL_SIZE" = "1"
    "DATABASE_MAX_OVERFLOW" = "1"
}

$warningSettings = @()

foreach ($entry in $forcedSettings.GetEnumerator()) {
    $current = [System.Environment]::GetEnvironmentVariable($entry.Key, "Process")
    if (-not [string]::IsNullOrWhiteSpace($current) -and $current -ne $entry.Value) {
        $warningSettings += $entry.Key
    }
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
}

foreach ($entry in $safeDefaults.GetEnumerator()) {
    $current = [System.Environment]::GetEnvironmentVariable($entry.Key, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}

$databaseUrl = [System.Environment]::GetEnvironmentVariable("DATABASE_URL", "Process")
if ([string]::IsNullOrWhiteSpace($databaseUrl)) {
    throw "DATABASE_URL no esta definido en $EnvFile."
}

$descriptorJson = & $pythonExe -c "import json, os; from sqlalchemy.engine import make_url; parsed = make_url(os.environ['DATABASE_URL']); print(json.dumps({'drivername': parsed.drivername, 'host': parsed.host or '', 'port': parsed.port or '', 'database': parsed.database or ''}))"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo interpretar DATABASE_URL."
}

$descriptor = $descriptorJson | ConvertFrom-Json

if ($warningSettings.Count -gt 0) {
    Write-Warning ("Se forzaron valores seguros para Supabase en: {0}" -f ($warningSettings -join ", "))
}

Write-Host "Backend root: $backendRoot"
Write-Host "Env file: $EnvFile"
Write-Host "Database driver: $($descriptor.drivername)"
Write-Host "Database host: $($descriptor.host)"
Write-Host "Database port: $($descriptor.port)"
Write-Host "Database name: $($descriptor.database)"
Write-Host "Schema management: $env:SCHEMA_MANAGEMENT_MODE"
Write-Host "Runtime bootstrap: $env:RUNTIME_BOOTSTRAP_ENABLED"
Write-Host "Knowledge autosync: $env:KNOWLEDGE_REPO_AUTOSYNC_ENABLED"

if ($CheckOnly) {
    Write-Host "Configuracion valida. No se inicio uvicorn porque se uso -CheckOnly."
    exit 0
}

$uvicornArgs = @(
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    $BindHost,
    "--port",
    $Port.ToString()
)

if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

& $pythonExe @uvicornArgs
exit $LASTEXITCODE
