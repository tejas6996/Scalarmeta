#requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"
$DOCKER_BUILD_TIMEOUT = 600
$PASS = 0

function Log([string]$msg) {
    $ts = (Get-Date).ToUniversalTime().ToString("HH:mm:ss")
    Write-Host "[$ts] $msg"
}

function PassMsg([string]$msg) {
    Write-Host ("[{0}] PASSED -- {1}" -f (Get-Date).ToUniversalTime().ToString("HH:mm:ss"), $msg) -ForegroundColor Green
    $script:PASS++
}

function FailMsg([string]$msg) {
    Write-Host ("[{0}] FAILED -- {1}" -f (Get-Date).ToUniversalTime().ToString("HH:mm:ss"), $msg) -ForegroundColor Red
}

function Hint([string]$msg) {
    Write-Host ("  Hint: {0}" -f $msg) -ForegroundColor Yellow
}

function StopAt([string]$step) {
    Write-Host ""
    Write-Host ("Validation stopped at {0}. Fix the above before continuing." -f $step) -ForegroundColor Red
    exit 1
}

function Run-ProcessWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()

    try {
        $proc = Start-Process -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -NoNewWindow `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru

        $finished = $proc.WaitForExit($TimeoutSec * 1000)
        if (-not $finished) {
            try { $proc.Kill() } catch {}
            return [pscustomobject]@{
                TimedOut  = $true
                ExitCode  = -1
                Output    = (Get-Content $stdout -Raw) + [Environment]::NewLine + (Get-Content $stderr -Raw)
            }
        }

        return [pscustomobject]@{
            TimedOut  = $false
            ExitCode  = $proc.ExitCode
            Output    = (Get-Content $stdout -Raw) + [Environment]::NewLine + (Get-Content $stderr -Raw)
        }
    }
    finally {
        Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
    }
}

try {
    $resolvedRepo = (Resolve-Path $RepoDir).Path
} catch {
    Write-Host ("Error: directory '{0}' not found" -f $RepoDir) -ForegroundColor Red
    exit 1
}

$PingUrl = $PingUrl.TrimEnd("/")

Write-Host ""
Write-Host "========================================" -ForegroundColor White
Write-Host "  OpenEnv Submission Validator (PS)" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Log ("Repo:     {0}" -f $resolvedRepo)
Log ("Ping URL: {0}" -f $PingUrl)
Write-Host ""

Log ("Step 1/3: Pinging HF Space ({0}/reset) ..." -f $PingUrl)

$httpCode = 0
try {
    $resp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$PingUrl/reset" -ContentType "application/json" -Body "{}" -TimeoutSec 30
    $httpCode = [int]$resp.StatusCode
} catch {
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
        $httpCode = [int]$_.Exception.Response.StatusCode.value__
    } else {
        $httpCode = 0
    }
}

if ($httpCode -eq 200) {
    PassMsg "HF Space is live and responds to /reset"
} elseif ($httpCode -eq 0) {
    FailMsg "HF Space not reachable (connection failed or timed out)"
    Hint "Check your network connection and that the Space is running."
    Hint ("Try: Invoke-WebRequest -Method Post -Uri '{0}/reset' -ContentType 'application/json' -Body '{{}}'" -f $PingUrl)
    StopAt "Step 1"
} else {
    FailMsg ("HF Space /reset returned HTTP {0} (expected 200)" -f $httpCode)
    Hint "Make sure your Space is running and the URL is correct."
    StopAt "Step 1"
}

Log "Step 2/3: Running docker build ..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    FailMsg "docker command not found"
    Hint "Install Docker Desktop and ensure docker is in PATH."
    StopAt "Step 2"
}

# Ensure the Docker daemon/engine is reachable before trying to build.
$dockerServerVersion = & docker version --format "{{.Server.Version}}" 2>&1
$dockerVersionExit = $LASTEXITCODE
if ($dockerVersionExit -ne 0 -or -not $dockerServerVersion) {
    FailMsg "Docker daemon is not reachable"
    Hint "Start Docker Desktop and wait until it reports 'Engine running'."
    Hint "In Docker Desktop, use Linux containers mode for this project."
    if ($dockerServerVersion) {
        $lines = ($dockerServerVersion | Out-String) -split "(`r`n|`n|`r)"
        $tail = $lines | Select-Object -Last 12
        $tail | ForEach-Object { Write-Host $_ }
    }
    StopAt "Step 2"
}
Log ("  Docker server version: {0}" -f (($dockerServerVersion | Out-String).Trim()))

$dockerContext = ""
if (Test-Path (Join-Path $resolvedRepo "Dockerfile")) {
    $dockerContext = $resolvedRepo
} elseif (Test-Path (Join-Path $resolvedRepo "server/Dockerfile")) {
    $dockerContext = (Join-Path $resolvedRepo "server")
} else {
    FailMsg "No Dockerfile found in repo root or server directory"
    StopAt "Step 2"
}

Log ("  Found Dockerfile in {0}" -f $dockerContext)

$build = Run-ProcessWithTimeout -FilePath "cmd.exe" -ArgumentList @("/c", "docker", "build", $dockerContext) -TimeoutSec $DOCKER_BUILD_TIMEOUT
$buildText = ($build.Output | Out-String)
$buildLooksSuccessful = (
    $buildText -match "(?im)^#\d+\s+DONE" -and
    $buildText -notmatch '(?im)error:|failed to solve|process ".*" did not complete successfully|denied|unauthorized'
)

if ($build.TimedOut) {
    FailMsg ("Docker build timed out (timeout={0}s)" -f $DOCKER_BUILD_TIMEOUT)
    $lines = ($buildText -split "(`r`n|`n|`r)")
    $tail = $lines | Select-Object -Last 20
    $tail | ForEach-Object { Write-Host $_ }
    StopAt "Step 2"
} elseif ($build.ExitCode -eq 0) {
    PassMsg "Docker build succeeded"
} elseif ($null -eq $build.ExitCode -and $buildLooksSuccessful) {
    PassMsg "Docker build succeeded (log-based detection)"
} else {
    $exitCodeDisplay = if ($null -eq $build.ExitCode) { "unknown" } else { [string]$build.ExitCode }
    FailMsg ("Docker build failed (exit code {0})" -f $exitCodeDisplay)
    $lines = ($buildText -split "(`r`n|`n|`r)")
    $tail = $lines | Select-Object -Last 20
    $tail | ForEach-Object { Write-Host $_ }
    StopAt "Step 2"
}

Log "Step 3/3: Running openenv validate ..."

if (-not (Get-Command openenv -ErrorAction SilentlyContinue)) {
    FailMsg "openenv command not found"
    Hint "Install it with: pip install openenv-core"
    StopAt "Step 3"
}

Push-Location $resolvedRepo
try {
    $validateOutput = & openenv validate 2>&1
    $validateExit = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($validateExit -eq 0) {
    PassMsg "openenv validate passed"
    if ($validateOutput) {
        Log ("  " + ($validateOutput -join [Environment]::NewLine + "  "))
    }
} else {
    FailMsg "openenv validate failed"
    if ($validateOutput) {
        $validateOutput | ForEach-Object { Write-Host $_ }
    }
    StopAt "Step 3"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor White
Write-Host "  All 3/3 checks passed!" -ForegroundColor Green
Write-Host "  Your submission is ready to submit." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor White
Write-Host ""

exit 0