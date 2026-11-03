# Deployment control plane for github-agent (Windows).
# Single authoritative implementation; the .bat wrappers all delegate here.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage.ps1 <action>
# Actions: status | start | stop | restart | ensure | verify | deploy | tunnel | preload | watchdog-register
param([Parameter(Position = 0)][string]$Action = "status")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Port = 8788
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$LogFile = Join-Path $Root "logs\agent.log"
$TunnelLog = Join-Path $Root "logs\cloudflared.log"
$TunnelErr = Join-Path $Root "logs\cloudflared.err.log"
$PublicUrlFile = Join-Path $Root "PUBLIC_URL.txt"
$Base = "http://127.0.0.1:$Port"

if (-not (Test-Path (Join-Path $Root "logs"))) { New-Item -ItemType Directory -Path (Join-Path $Root "logs") | Out-Null }

function Get-Health {
    try { return Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 4 } catch { return $null }
}

function Get-ApiPid {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return $conn.OwningProcess }
    return $null
}

function Wait-Healthy([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $h = Get-Health
        if ($null -ne $h -and $h.status -eq "ok") { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Stop-Api {
    $pid2 = Get-ApiPid
    if ($null -eq $pid2) { Write-Host "API not running on $Port."; return }
    Write-Host "Stopping API process (pid $pid2) ..."
    try { Stop-Process -Id $pid2 -Force -ErrorAction Stop } catch { Write-Host "Stop-Process failed: $_" }
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline -and (Get-ApiPid)) { Start-Sleep -Seconds 1 }
}

function Start-Api {
    if (-not (Test-Path $Py)) {
        Write-Host "[ERROR] venv python missing. Run: scripts\setup_windows.bat"; exit 1
    }
    if (Get-Health) { Write-Host "API already healthy on $Base."; return }
    # refuse obviously-insecure defaults before exposing the service
    $envFile = Join-Path $Root ".env"
    if (Test-Path $envFile) {
        $envText = Get-Content $envFile -Raw
        if ($envText -match "SECRET_KEY=please-change-me" -or $envText -match "ADMIN_TOKEN=change-me-admin-token") {
            Write-Host "[WARN] .env still contains default secrets - rotate them before public exposure."
        }
    }
    # preloaded models keep the first request from paying model load
    $env:TRUST_PROXY_HEADERS = "true"
    Start-Process -FilePath $Py `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$Port" `
        -WorkingDirectory $Root -WindowStyle Minimized
    Write-Host "API process launched (loading models, this can take ~60s) ..."
}

function Ensure-Tunnel {
    $cf = Get-Process -Name cloudflared -ErrorAction SilentlyContinue
    if ($null -eq $cf) {
        $exe = Join-Path $Root "tools\cloudflared.exe"
        if (-not (Test-Path $exe)) { Write-Host "[WARN] tools\cloudflared.exe not found; tunnel skipped."; return }
        Write-Host "Starting cloudflared tunnel ..."
        Start-Process -FilePath $exe -ArgumentList "tunnel", "--url", "http://localhost:$Port" `
            -WorkingDirectory $Root -WindowStyle Hidden `
            -RedirectStandardOutput $TunnelLog -RedirectStandardError $TunnelErr
        Start-Sleep -Seconds 10
    }
    $url = $null
    $deadline = (Get-Date).AddSeconds(25)
    while ((Get-Date) -lt $deadline -and -not $url) {
        foreach ($f in @($TunnelLog, $TunnelErr)) {
            if (Test-Path $f) {
                $m = Select-String -Path $f -Pattern "https://[a-z0-9-]*\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -Last 1
                if ($m) { $url = $m.Matches[0].Value; break }
            }
        }
        if (-not $url) { Start-Sleep -Seconds 5 }
    }
    if ($url) {
        Set-Content -Path $PublicUrlFile -Value $url
        Write-Host "Tunnel URL: $url (saved to PUBLIC_URL.txt)"
    } else {
        Write-Host "[WARN] Tunnel URL not found yet; check $TunnelErr"
    }
}

function Invoke-Verify {
    $fail = 0
    $h = Get-Health
    if ($null -ne $h -and $h.status -eq "ok") { Write-Host "[PASS] /health" } else { Write-Host "[FAIL] /health"; $fail++ }
    try {
        $info = Invoke-RestMethod -Uri "$Base/info" -TimeoutSec 5
        Write-Host "[PASS] /info version=$($info.version)"
    } catch { Write-Host "[FAIL] /info"; $fail++ }
    try {
        $body = '{"query": "terminal file manager", "top_k": 2}'
        $r = Invoke-RestMethod -Uri "$Base/api/recommend" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
        if ($r.results.Count -gt 0) { Write-Host "[PASS] /api/recommend ($($r.results.Count) hits)" }
        else { Write-Host "[FAIL] /api/recommend returned no hits"; $fail++ }
    } catch { Write-Host "[FAIL] /api/recommend: $_"; $fail++ }
    try {
        $body = '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'
        $r = Invoke-RestMethod -Uri "$Base/mcp" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10
        if ($r.result.protocolVersion) { Write-Host "[PASS] /mcp initialize" }
        else { Write-Host "[FAIL] /mcp initialize"; $fail++ }
    } catch { Write-Host "[FAIL] /mcp initialize: $_"; $fail++ }
    if ($fail -gt 0) { Write-Host "VERIFY FAILED ($fail checks)"; exit 1 }
    Write-Host "VERIFY PASSED"
}

switch ($Action) {
    "status" {
        $h = Get-Health
        $apiPid = Get-ApiPid
        if ($null -ne $h -and $h.status -eq "ok") {
            Write-Host "API: RUNNING (pid $apiPid) v$((Invoke-RestMethod -Uri "$Base/info" -TimeoutSec 5).version)"
            Write-Host "  llm=$($h.components.llm_provider) embedder=$($h.components.embedder_provider) store=$($h.components.vector_store) scheduler=$($h.components.scheduler_running)"
        } elseif ($apiPid) {
            Write-Host "API: STARTING/UNHEALTHY (pid $apiPid, port bound but /health failing)"
        } else {
            Write-Host "API: STOPPED"
        }
        $tunnel = Get-Process -Name cloudflared -ErrorAction SilentlyContinue
        if ($tunnel -and (Test-Path $PublicUrlFile)) {
            Write-Host "Tunnel: RUNNING ($(Get-Content $PublicUrlFile -First 1))"
        } elseif ($tunnel) { Write-Host "Tunnel: RUNNING (URL not captured yet)" }
        else { Write-Host "Tunnel: STOPPED" }
    }
    "start" { Start-Api; if (Wait-Healthy 150) { Write-Host "API is healthy on $Base" } else { Write-Host "[ERROR] API not healthy within 150s - tail of $LogFile :"; if (Test-Path $LogFile) { Get-Content $LogFile -Tail 15 }; exit 1 } }
    "stop" { Stop-Api }
    "restart" { Stop-Api; Start-Api; if (Wait-Healthy 150) { Write-Host "API restarted and healthy" } else { Write-Host "[ERROR] not healthy after restart"; exit 1 } }
    "ensure" {
        if (Get-Health) { Write-Host "ensure: API already healthy." }
        else { Stop-Api; Start-Api; if (-not (Wait-Healthy 150)) { Write-Host "[ERROR] watchdog could not recover API"; exit 1 }; Write-Host "ensure: API recovered." }
        Ensure-Tunnel
    }
    "verify" { Invoke-Verify }
    "deploy" {
        Push-Location $Root
        git pull --ff-only; if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "[ERROR] git pull failed"; exit 1 }
        & $Py -m pip install --quiet -r requirements.txt; if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "[ERROR] pip sync failed"; exit 1 }
        & $Py -m pytest -q; if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "[ERROR] test gate failed - NOT restarting"; exit 1 }
        Pop-Location
        Stop-Api; Start-Api
        if (-not (Wait-Healthy 150)) { Write-Host "[ERROR] not healthy after deploy"; exit 1 }
        Invoke-Verify
    }
    "tunnel" { Ensure-Tunnel }
    "preload" {
        $env:HF_ENDPOINT = "https://hf-mirror.com"
        & $Py -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.pt','pytorch_model.bin']); print('bge-m3 cached')"
        & $Py -c "from laya import Router; Router(preload=True); print('laya checkpoints cached')"
    }
    "watchdog-register" {
        $task = "GitHubAgentWatchdog"
        $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Root\scripts\manage.ps1`" ensure"
        schtasks /create /tn $task /tr $cmd /sc minute /mo 5 /rl limited /f | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "Watchdog registered: every 5 min runs 'manage.ps1 ensure' (self-heal API + tunnel)." }
        else { Write-Host "[ERROR] schtasks registration failed"; exit 1 }
    }
    default { Write-Host "Unknown action '$Action'. Use: status start stop restart ensure verify deploy tunnel preload watchdog-register"; exit 1 }
}
