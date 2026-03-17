# Load environment variables from .env file for migration testing
# Usage: . .\crates\velesdb-migrate\scripts\load-env.ps1

param(
    [string]$EnvFile = ".env"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))

# Try to find .env file
$envPath = if (Test-Path $EnvFile) { 
    $EnvFile 
} elseif (Test-Path (Join-Path $projectRoot ".env")) {
    Join-Path $projectRoot ".env"
} else {
    Write-Output "No .env file found"
    Write-Output "   Create one from .env.example:"
    Write-Output "   Copy-Item .env.example .env"
    return
}

Write-Output "Loading environment from: $envPath"

# Parse and set environment variables
Get-Content $envPath | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        if ($value -and $value -ne "your-service-role-key" -and $value -notlike "*YOUR_PROJECT*") {
            Set-Item -Path "env:$name" -Value $value
            Write-Output "   Set: $name"
        }
    }
}

Write-Output ""
Write-Output "Ready to test! Run:"
Write-Output "   cargo test -p velesdb-migrate --test integration_test -- --ignored --nocapture"
Write-Output "   cargo bench -p velesdb-migrate"
