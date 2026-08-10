param(
    [ValidateSet("smoke", "unit", "api", "api-full", "full")]
    [string]$Suite = "smoke"
)

$backendRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $backendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "No se encontro $pythonPath. Crea primero el entorno virtual con 'python -m venv .venv' e instala requirements."
}

$markerExpression = switch ($Suite) {
    "smoke" { "smoke" }
    "unit" { "unit and not slow" }
    "api" { "api and not slow" }
    "api-full" { "api" }
    "full" { "" }
}

Push-Location $backendRoot
try {
    if ($markerExpression) {
        & $pythonPath -m pytest tests -q -m $markerExpression
    }
    else {
        & $pythonPath -m pytest tests -q
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
