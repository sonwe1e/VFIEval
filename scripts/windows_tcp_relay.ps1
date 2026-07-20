[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("install", "status", "remove", "self-test")]
    [string]$Action,

    [string]$ListenAddress = "0.0.0.0",
    [ValidateRange(1, 65535)]
    [int]$ListenPort = 8765,
    [string]$TargetAddress,
    [ValidateRange(1, 65535)]
    [int]$TargetPort = 8765,
    [string]$BaseUrl
)

$ErrorActionPreference = "Stop"
$FirewallRule = "VFIEval Relay TCP $ListenPort"

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Action '$Action' requires an elevated PowerShell window."
    }
}

function Resolve-RelayTarget([string]$Address) {
    if ([string]::IsNullOrWhiteSpace($Address)) {
        throw "-TargetAddress is required for install."
    }
    $parsed = $null
    if ([Net.IPAddress]::TryParse($Address, [ref]$parsed) -and $parsed.AddressFamily -eq "InterNetwork") {
        return $parsed.IPAddressToString
    }
    $resolved = [Net.Dns]::GetHostAddresses($Address) |
        Where-Object { $_.AddressFamily -eq "InterNetwork" } |
        Select-Object -First 1
    if ($null -eq $resolved) {
        throw "Target '$Address' did not resolve to an IPv4 address."
    }
    return $resolved.IPAddressToString
}

function Invoke-NetshChecked([string[]]$Arguments) {
    & netsh @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "netsh failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Remove-RelayRule {
    & netsh interface portproxy delete v4tov4 "listenaddress=$ListenAddress" "listenport=$ListenPort" | Out-Host
    $rules = (& netsh interface portproxy show v4tov4 | Out-String)
    $pattern = "(?m)^\s*" + [regex]::Escape($ListenAddress) + "\s+" + $ListenPort + "\s+"
    if ($rules -match $pattern) {
        throw "The TCP relay ${ListenAddress}:${ListenPort} could not be removed."
    }
}

function Test-HttpEndpoint([string]$Name, [string]$Uri, [hashtable]$Headers = @{}) {
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 15 -Headers $Headers
        [pscustomobject]@{
            Check = $Name
            Status = "ok"
            HttpStatus = [int]$response.StatusCode
            Uri = $Uri
            Detail = ""
        }
    }
    catch {
        $statusCode = $null
        if ($null -ne $_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
        [pscustomobject]@{
            Check = $Name
            Status = "failed"
            HttpStatus = $statusCode
            Uri = $Uri
            Detail = $_.Exception.Message
        }
    }
}

switch ($Action) {
    "install" {
        Assert-Administrator
        $targetIp = Resolve-RelayTarget $TargetAddress
        Remove-RelayRule
        Invoke-NetshChecked @(
            "interface", "portproxy", "add", "v4tov4",
            "listenaddress=$ListenAddress", "listenport=$ListenPort",
            "connectaddress=$targetIp", "connectport=$TargetPort"
        )
        Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule
        New-NetFirewallRule `
            -DisplayName $FirewallRule `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort $ListenPort `
            -Profile Private,Domain | Out-Null
        Write-Host "VFIEval relay installed: ${ListenAddress}:${ListenPort} -> ${targetIp}:${TargetPort}"
        Write-Host "Run: .\scripts\windows_tcp_relay.ps1 self-test -ListenPort $ListenPort"
    }
    "status" {
        Write-Host "Configured Windows TCP relays:"
        & netsh interface portproxy show v4tov4
        Write-Host "`nListener state for TCP ${ListenPort}:"
        Get-NetTCPConnection -State Listen -LocalPort $ListenPort -ErrorAction SilentlyContinue |
            Select-Object LocalAddress, LocalPort, OwningProcess
        Write-Host "`nFirewall rule:"
        Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue |
            Select-Object DisplayName, Enabled, Direction, Action, Profile
    }
    "remove" {
        Assert-Administrator
        Remove-RelayRule
        Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule
        Write-Host "VFIEval relay and its firewall rule were removed for ${ListenAddress}:${ListenPort}."
    }
    "self-test" {
        if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
            $BaseUrl = "http://127.0.0.1:$ListenPort"
        }
        $BaseUrl = $BaseUrl.TrimEnd("/")
        $checks = @(
            Test-HttpEndpoint "homepage" "$BaseUrl/"
            Test-HttpEndpoint "health" "$BaseUrl/api/health"
            Test-HttpEndpoint "blind-page" "$BaseUrl/evaluate/relay-self-test"
            Test-HttpEndpoint "byte-range" "$BaseUrl/blind.js" @{ Range = "bytes=0-15" }
        )
        $checks | Format-Table -AutoSize
        $failed = @($checks | Where-Object { $_.Status -ne "ok" })
        if ($failed.Count -gt 0) {
            throw "$($failed.Count) relay self-test check(s) failed."
        }
        $health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 15
        Write-Host "VFIEval relay self-test passed. Build: $($health.release.build_id), uptime: $([math]::Round($health.release.uptime_seconds, 1))s"
    }
}
