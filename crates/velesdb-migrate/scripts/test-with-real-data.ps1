# VelesDB Migration - Test with Real Data
# Usage: .\scripts\test-with-real-data.ps1
#
# Environment variables required:
#   $env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
#   $env:SUPABASE_SERVICE_KEY = "your-service-key"
#   $env:SUPABASE_TABLE = "your_table_name"

param(
    [switch]$IntegrationTests,
    [switch]$Benchmarks,
    [switch]$FullMigration,
    [switch]$All
)

$ErrorActionPreference = "Stop"

Write-Output "================================================================"
Write-Output "         VelesDB Migration - Real Data Testing"
Write-Output "================================================================"
Write-Output ""

# Check environment variables
if (-not $env:SUPABASE_URL) {
    Write-Output "SUPABASE_URL not set"
    Write-Output "   Set it with: `$env:SUPABASE_URL = 'https://YOUR_PROJECT.supabase.co'"
    exit 1
}

if (-not $env:SUPABASE_SERVICE_KEY) {
    Write-Output "SUPABASE_SERVICE_KEY not set"
    Write-Output "   Set it with: `$env:SUPABASE_SERVICE_KEY = 'your-service-key'"
    exit 1
}

if (-not $env:SUPABASE_TABLE) {
    Write-Output "SUPABASE_TABLE not set"
    Write-Output "   Set it with: `$env:SUPABASE_TABLE = 'your_table_name'"
    exit 1
}
$table = $env:SUPABASE_TABLE
Write-Output "Environment configured:"
Write-Output "   URL: $($env:SUPABASE_URL)"
Write-Output "   Table: $table"
Write-Output ""

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptPath)
Set-Location $projectRoot

if ($All) {
    $IntegrationTests = $true
    $Benchmarks = $true
    $FullMigration = $true
}

# 1. Run Integration Tests
if ($IntegrationTests) {
    Write-Output "==============================================================="
    Write-Output "Running Integration Tests..."
    Write-Output "==============================================================="
    Write-Output ""

    cargo test -p velesdb-migrate --test integration_test -- --ignored --nocapture

    if ($LASTEXITCODE -eq 0) {
        Write-Output ""
        Write-Output "Integration tests passed!"
    } else {
        Write-Output ""
        Write-Output "Integration tests failed!"
        exit 1
    }
    Write-Output ""
}

# 2. Run Benchmarks
if ($Benchmarks) {
    Write-Output "==============================================================="
    Write-Output "Running Benchmarks..."
    Write-Output "==============================================================="
    Write-Output ""

    cargo bench -p velesdb-migrate

    Write-Output ""
    Write-Output "Benchmarks completed! Results in target/criterion/"
    Write-Output ""
}

# 3. Full Migration Test
if ($FullMigration) {
    Write-Output "==============================================================="
    Write-Output "Running Full Migration Test..."
    Write-Output "==============================================================="
    Write-Output ""

    # Create temp directory for test
    $testDir = Join-Path $env:TEMP "velesdb_migration_test_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $testDir -Force | Out-Null

    Write-Output "Test directory: $testDir"
    Write-Output ""

    # Step 1: Detect schema
    Write-Output "1. Detecting schema..."
    $configFile = Join-Path $testDir "migration.yaml"
    
    & .\target\release\velesdb-migrate.exe detect `
        --source supabase `
        --url $env:SUPABASE_URL `
        --collection $table `
        --api-key $env:SUPABASE_SERVICE_KEY `
        --output $configFile `
        --dest-path (Join-Path $testDir "velesdb_data")
    
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Schema detection failed!"
        exit 1
    }

    Write-Output ""
    Write-Output "Generated config:"
    Get-Content $configFile | Write-Output
    Write-Output ""

    # Step 2: Validate config
    Write-Output "2. Validating configuration..."
    & .\target\release\velesdb-migrate.exe validate --config $configFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Validation failed!"
        exit 1
    }
    Write-Output "Configuration valid!"
    Write-Output ""

    # Step 3: Show schema
    Write-Output "3. Fetching source schema..."
    & .\target\release\velesdb-migrate.exe schema --config $configFile
    Write-Output ""

    # Step 4: Dry run
    Write-Output "4. Dry run (no data written)..."
    & .\target\release\velesdb-migrate.exe run --config $configFile --dry-run
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Dry run successful!"
    } else {
        Write-Output "WARNING: Dry run had issues"
    }
    Write-Output ""

    # Ask before actual migration
    Write-Output "==============================================================="
    $confirm = Read-Host "Run actual migration? This will import data to $testDir/velesdb_data (y/N)"
    
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Write-Output ""
        Write-Output "5. Running migration..."
        
        $startTime = Get-Date
        & .\target\release\velesdb-migrate.exe run --config $configFile
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Output ""
            Write-Output "Migration completed in $($duration.TotalSeconds) seconds!"
            Write-Output ""
            Write-Output "Data stored in: $testDir\velesdb_data"

            # Show file sizes
            $dataPath = Join-Path $testDir "velesdb_data"
            if (Test-Path $dataPath) {
                $size = (Get-ChildItem $dataPath -Recurse | Measure-Object -Property Length -Sum).Sum
                $sizeMB = [math]::Round($size / 1MB, 2)
                Write-Output "Total size: $sizeMB MB"
            }
        } else {
            Write-Output "Migration failed!"
        }
    } else {
        Write-Output "Skipping actual migration"
    }

    Write-Output ""
    Write-Output "Test directory: $testDir"
    Write-Output "   (delete manually when done testing)"
}

Write-Output ""
Write-Output "================================================================"
Write-Output "                    Testing Complete!"
Write-Output "================================================================"
