<#
.SYNOPSIS
    Run VelesDB benchmarks with reproducible settings.
    
.DESCRIPTION
    Executes benchmarks and exports results to JSON/CSV.
    EPIC-026/US-001: Reproducible benchmark protocol.
    
.PARAMETER Dataset
    Dataset to use: random10k, random100k, random1m
    
.PARAMETER Runs
    Number of benchmark runs (default: 5)
    
.PARAMETER Output
    Output directory for results
    
.PARAMETER Quick
    Skip warmup and use fewer samples (for quick validation)
    
.EXAMPLE
    .\bench_run.ps1 -Dataset random100k -Runs 5 -Output ./results
#>

param(
    [ValidateSet("random10k", "random100k", "random1m")]
    [string]$Dataset = "random10k",
    
    [int]$Runs = 5,
    
    [string]$Output = "./results",
    
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Environment Info Collection
# =============================================================================

function Get-EnvironmentInfo {
    $rustVersion = (rustc --version 2>$null) -replace "rustc ", ""
    $cargoVersion = (cargo --version 2>$null) -replace "cargo ", ""
    
    # Get VelesDB version from Cargo.toml
    $cargoToml = Get-Content -Path "Cargo.toml" -Raw
    $versionMatch = [regex]::Match($cargoToml, 'version\s*=\s*"([^"]+)"')
    $velesdbVersion = if ($versionMatch.Success) { $versionMatch.Groups[1].Value } else { "unknown" }
    
    @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        os = [System.Environment]::OSVersion.ToString()
        os_version = [System.Environment]::OSVersion.Version.ToString()
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
        cpu_cores = [Environment]::ProcessorCount
        ram_gb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
        rust_version = $rustVersion
        cargo_version = $cargoVersion
        velesdb_version = $velesdbVersion
        features = "simd"
        dataset = $Dataset
    }
}

# =============================================================================
# Criterion Output Parser
# =============================================================================

function Parse-CriterionOutput {
    param([string[]]$Output)
    
    $metrics = @{
        benchmarks = @{}
    }
    
    $currentBench = $null
    
    foreach ($line in $Output) {
        # Match benchmark name: "Benchmarking name"
        if ($line -match "Benchmarking\s+(.+)$") {
            $currentBench = $Matches[1].Trim()
        }
        
        # Match time result: "time:   [1.2345 µs 1.3456 µs 1.4567 µs]"
        if ($line -match "time:\s+\[([0-9.]+)\s*([µμnm]?s)\s+([0-9.]+)\s*([µμnm]?s)\s+([0-9.]+)\s*([µμnm]?s)\]") {
            $p50Value = [double]$Matches[1]
            $p50Unit = $Matches[2]
            $meanValue = [double]$Matches[3]
            $p99Value = [double]$Matches[5]
            
            # Convert to microseconds
            $multiplier = switch -Regex ($p50Unit) {
                "ns" { 0.001 }
                "[µμ]s" { 1 }
                "ms" { 1000 }
                "s" { 1000000 }
                default { 1 }
            }
            
            if ($currentBench) {
                $metrics.benchmarks[$currentBench] = @{
                    p50_us = [math]::Round($p50Value * $multiplier, 3)
                    mean_us = [math]::Round($meanValue * $multiplier, 3)
                    p99_us = [math]::Round($p99Value * $multiplier, 3)
                }
            }
        }
        
        # Match throughput: "thrpt:  [1234.5 elem/s 1345.6 elem/s 1456.7 elem/s]"
        if ($line -match "thrpt:\s+\[([0-9.]+)\s*(\w+/s)") {
            $throughput = [double]$Matches[1]
            if ($currentBench -and $metrics.benchmarks[$currentBench]) {
                $metrics.benchmarks[$currentBench]["throughput"] = [math]::Round($throughput, 2)
            }
        }
    }
    
    return $metrics
}

# =============================================================================
# Main Execution
# =============================================================================

Write-Output "==============================================================="
Write-Output "  VelesDB Reproducible Benchmark Suite (EPIC-026/US-001)"
Write-Output "==============================================================="

# Collect environment info
Write-Output "`nCollecting environment information..."
$envInfo = Get-EnvironmentInfo

Write-Output "  OS: $($envInfo.os)"
Write-Output "  CPU: $($envInfo.cpu) ($($envInfo.cpu_cores) cores)"
Write-Output "  RAM: $($envInfo.ram_gb) GB"
Write-Output "  Rust: $($envInfo.rust_version)"
Write-Output "  VelesDB: $($envInfo.velesdb_version)"

# Build release
Write-Output "`nBuilding release with SIMD features..."
cargo build --release --features "simd" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "Build failed!"
    exit 1
}
Write-Output "  Build successful"

# Warmup (unless Quick mode)
if (-not $Quick) {
    Write-Output "`nWarmup run (discarded)..."
    cargo bench --bench hnsw_bench -- --warm-up-time 3 --noplot 2>&1 | Out-Null
}

# Run benchmarks
Write-Output "`nRunning $Runs benchmark iterations..."

$allResults = @()
$sampleSize = if ($Quick) { 10 } else { 50 }
$warmupTime = if ($Quick) { 1 } else { 3 }

for ($i = 1; $i -le $Runs; $i++) {
    Write-Output "  Run $i/$Runs..."
    
    $output = cargo bench --bench hnsw_bench -- --sample-size $sampleSize --warm-up-time $warmupTime --noplot 2>&1
    $metrics = Parse-CriterionOutput $output
    
    if ($metrics.benchmarks.Count -gt 0) {
        $allResults += $metrics
    }
}

# Aggregate results
Write-Output "`nAggregating results..."

$aggregated = @{
    environment = $envInfo
    runs = $Runs
    quick_mode = $Quick.IsPresent
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    benchmarks = @{}
}

# Collect all benchmark names
$benchNames = @()
foreach ($run in $allResults) {
    $benchNames += $run.benchmarks.Keys
}
$benchNames = $benchNames | Select-Object -Unique

foreach ($name in $benchNames) {
    $p50Values = @()
    $meanValues = @()
    $p99Values = @()
    
    foreach ($run in $allResults) {
        if ($run.benchmarks[$name]) {
            $p50Values += $run.benchmarks[$name].p50_us
            $meanValues += $run.benchmarks[$name].mean_us
            $p99Values += $run.benchmarks[$name].p99_us
        }
    }
    
    if ($p50Values.Count -gt 0) {
        $aggregated.benchmarks[$name] = @{
            p50_us = [math]::Round(($p50Values | Measure-Object -Average).Average, 3)
            mean_us = [math]::Round(($meanValues | Measure-Object -Average).Average, 3)
            p99_us = [math]::Round(($p99Values | Measure-Object -Average).Average, 3)
            std_dev_us = [math]::Round(($meanValues | Measure-Object -StandardDeviation).StandardDeviation, 3)
            samples = $p50Values.Count
        }
    }
}

# Create output directory
$outputDir = New-Item -ItemType Directory -Path $Output -Force

# Export results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$jsonPath = Join-Path $outputDir "benchmark_$timestamp.json"
$latestPath = Join-Path $outputDir "latest.json"

$aggregated | ConvertTo-Json -Depth 10 | Out-File $jsonPath -Encoding UTF8
$aggregated | ConvertTo-Json -Depth 10 | Out-File $latestPath -Encoding UTF8

# Display results
Write-Output "`n==============================================================="
Write-Output "  Results Summary"
Write-Output "==============================================================="

foreach ($name in $aggregated.benchmarks.Keys | Sort-Object) {
    $bench = $aggregated.benchmarks[$name]
    Write-Output "`n  $name"
    Write-Output "    p50: $($bench.p50_us) us | mean: $($bench.mean_us) us | p99: $($bench.p99_us) us"
}

Write-Output "`nResults saved to:"
Write-Output "   $jsonPath"
Write-Output "   $latestPath"

Write-Output "`nBenchmark complete!"
