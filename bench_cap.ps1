<#  bench_cap.ps1 - qwen38-zig only: sweep expert-cap x threads, all WARM.
    No cache flush - the OS page cache and .colizig_usage priors stay hot, so
    each row is steady-state decode. One warmup run per config, then measure. #>
param(
  # Path to the Qwen3.8-Flash-Next-FP8 checkpoint dir (or set $env:QWEN38_MODEL).
  [string]$Model  = $(if ($env:QWEN38_MODEL) { $env:QWEN38_MODEL } else { "D:\Models\Qwen38-FP8" }),
  [string]$Prompt = "The capital of France is",
  [int]   $Tokens = 24,
  [string]$Zig    = (Join-Path $PSScriptRoot "zig-out\bin\qwen38-zig.exe")
)

$tk  = (& $Zig tokenize $Model --prompt $Prompt) 2>&1 | Out-String
$ids = ([regex]::Match($tk,'ids \(\d+\): ([\d ]+)').Groups[1].Value).Trim() -replace '\s+',','
Write-Host "prompt ids: $ids   ($Tokens new tokens, warm)" -ForegroundColor Cyan

function Run($cap,$threads) {
  $a = @("forward",$Model,"--tokens",$ids,"--steps","$Tokens","--expert-cap","$cap","--threads","$threads","--ram-limit","28G")
  & $Zig @a *> $null                       # warmup
  $o = Join-Path $env:TEMP "cap_out.txt"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $Zig -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError ($o+".e")
  $peak = 0.0
  while (-not $p.HasExited) { try { $p.Refresh(); $w=$p.WorkingSet64/1GB; if($w -gt $peak){$peak=$w} } catch {}; Start-Sleep -Milliseconds 400 }
  $sw.Stop()
  $t = (Get-Content $o -Raw) + "`n" + (Get-Content ($o+".e") -Raw)
  function M($rx){ $m=[regex]::Match($t,$rx); if($m.Success){$m.Groups[1].Value}else{"?"} }
  [pscustomobject]@{
    cap=$cap; threads=$threads
    ttft=(M 'TTFT:\s*([\d.]+)\s*s'); decode=(M 'decode:\s*([\d.]+)\s*tok/s')
    demand=(M 'expert demand loads:\s*(\d+)'); load=(M 'model load:\s*([\d.]+)\s*s')
    wall=[math]::Round($sw.Elapsed.TotalSeconds,1); peakGB=[math]::Round($peak,1)
  }
}

$rows = @()
foreach ($cfg in @(@(64,12),@(64,6),@(512,12),@(512,6))) {
  Write-Host "  running cap $($cfg[0]) / $($cfg[1]) threads ..." -ForegroundColor DarkGray
  $rows += Run $cfg[0] $cfg[1]
}
Write-Host ""
$rows | Format-Table cap,threads,load,ttft,decode,demand,wall,peakGB -AutoSize
