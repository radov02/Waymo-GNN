param(
    [string]$EnvFile = (Join-Path $PSScriptRoot '.env')
)

if (-not (Test-Path $EnvFile)) {
    Write-Error "Env file not found: $EnvFile"
    exit 1
}

# load simple KEY=VALUE .env (ignore lines starting with # or //)
Get-Content $EnvFile | ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not ($_ -match '^\s*#') -and -not ($_ -match '^\s*//') } |
    ForEach-Object {
        if ($_ -match '^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$') {
            $k = $matches[1]
            $v = $matches[2].Trim()
            # strip inline comments starting with # or //
            $v = $v -replace '\s+#.*$',''
            $v = $v -replace '\s+//.*$',''
            $v = $v.Trim()
            if ($v -match '^"(.*)"$' -or $v -match "^'(.*)'$") { $v = $matches[1] }
            Set-Variable -Name $k -Value $v -Scope Script
        }
    }

# required vars
$required = 'REMOTE_HOST_IP','REMOTE_SSH_PORT','REMOTE_SSH_USER','PATH_TO_SSH_KEY','REMOTE_PATH_VISUALIZATIONS','PATH_TO_DOWNLOAD_TO'
$missing = $required | Where-Object { -not (Get-Variable -Name $_ -ErrorAction SilentlyContinue) }
if ($missing) {
    Write-Error "Missing required env vars: $($missing -join ', ')"
    exit 1
}

if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Error "scp not found. Install OpenSSH client or ensure scp is in PATH."
    exit 1
}

# ensure local directory exists
New-Item -ItemType Directory -Force -Path $PATH_TO_DOWNLOAD_TO | Out-Null

$remote = "${REMOTE_SSH_USER}@${REMOTE_HOST_IP}:`"$REMOTE_PATH_VISUALIZATIONS`""
$args = @('-r', '-P', $REMOTE_SSH_PORT, '-i', $PATH_TO_SSH_KEY, $remote, $PATH_TO_DOWNLOAD_TO)

Write-Output "Running: scp $($args -join ' ')"
& scp @args
if ($LASTEXITCODE -ne 0) {
    Write-Error "scp failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output "Download completed: $REMOTE_PATH_VISUALIZATIONS -> $PATH_TO_DOWNLOAD_TO"
