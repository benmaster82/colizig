<#
  bench_ab.ps1 - fair A/B: colibri vs qwen38-zig on the same real checkpoint.

  Same prompt, same generated-token count, same expert-cache cap, same thread
  count. Each engine starts with a cold OS page cache (RAM ballast) and no
  learned expert priors (.coli_usage removed). Both are one-shot processes
  (no warm server), both greedy, both use the checkpoint's own tokenizer.json.

  Usage:
    powershell -ExecutionPolicy Bypass -File bench_ab.ps1
    powershell -ExecutionPolicy Bypass -File bench_ab.ps1 -Prompt "..." -Tokens 32 -Cap 64 -Threads 12
#>
param(
  # Path to the Qwen3.8-Flash-Next-FP8 checkpoint dir (or set $env:QWEN38_MODEL).
  [string]$Model   = $(if ($env:QWEN38_MODEL) { $env:QWEN38_MODEL } else { "D:\Models\Qwen38-FP8" }),
  [string]$Prompt  = "The capital of France is",
  [int]   $Tokens  = 32,
  [int]   $Cap     = 64,
  [int]   $Threads = [Environment]::ProcessorCount,
  [string]$Zig     = (Join-Path $PSScriptRoot "zig-out\bin\qwen38-zig.exe"),
  # Optional: an OpenMP build of colibri's qwen38 engine for the A/B, plus its MinGW runtime dir.
  [string]$Colibri = (Join-Path $PSScriptRoot "_colibri\colibri-main\c\qwen38.exe"),
  [string]$MinGW   = "C:\msys64\mingw64\bin"
)

function FlushCache {
  Write-Host "  flushing OS page cache ..." -ForegroundColor DarkGray
  $blocks = New-Object System.Collections.ArrayList
  try {
    for ($i = 0; $i -lt 22; $i++) {
      $b = New-Object byte[] (1GB)
      for ($j = 0; $j -lt $b.Length; $j += 4096) { $b[$j] = 1 }
      [void]$blocks.Add($b)
    }
  } catch {}
  $blocks.Clear(); [System.GC]::Collect(); Start-Sleep 1
}

function RunExe([string]$exe, [string[]]$argv, [hashtable]$envv) {
  foreach ($k in $envv.Keys) { Set-Item "Env:$k" $envv[$k] }
  $o = Join-Path $env:TEMP "ab_out.txt"; $e = Join-Path $env:TEMP "ab_err.txt"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru `
        -RedirectStandardOutput $o -RedirectStandardError $e
  # poll the child's peak working set while it runs (PeakWorkingSet64 is gone
  # once the process exits)
  $peak = 0.0
  while (-not $p.HasExited) {
    try { $p.Refresh(); $ws = $p.WorkingSet64 / 1GB; if ($ws -gt $peak) { $peak = $ws } } catch {}
    Start-Sleep -Milliseconds 500
  }
  $sw.Stop()
  $so = (Get-Content $o -Raw -EA SilentlyContinue); $se = (Get-Content $e -Raw -EA SilentlyContinue)
  [pscustomobject]@{ Out = "$so"; Err = "$se"; Text = "$so`n$se"; Wall = $sw.Elapsed.TotalSeconds; PeakGB = [math]::Round($peak,1); Exit = $p.ExitCode }
}
function M([string]$text, [string]$rx) { $m = [regex]::Match($text, $rx); if ($m.Success) { $m.Groups[1].Value } else { "?" } }

# --- resolve prompt -> token ids with qwen38-zig's tokenizer (== colibri's) ---
$tk = (& $Zig tokenize $Model --prompt $Prompt) 2>&1 | Out-String
$ids = (M $tk 'ids \(\d+\): ([\d ]+)').Trim() -replace '\s+', ','
if ($ids -eq '?' -or -not $ids) { throw "could not tokenize the prompt" }
$promptCount = ($ids -split ',').Count
Write-Host ""
Write-Host "prompt : `"$Prompt`"  ->  ids: $ids" -ForegroundColor Cyan
Write-Host "config : $Tokens new tokens - cap $Cap/layer - $Threads threads - greedy - cold start" -ForegroundColor Cyan

$promptFile = Join-Path $env:TEMP "ab_prompt.txt"
[System.IO.File]::WriteAllText($promptFile, $Prompt)   # no trailing newline

$haveColibri = Test-Path $Colibri
$cLoad = "-"; $cTTFT = "-"; $cSpeed = "-"; $cHit = "-"; $cText = ""; $cWall = 0.0; $cPeak = "-"

# ============================== colibri ==============================
if ($haveColibri) {
  Write-Host "`n=== colibri ===" -ForegroundColor Yellow
  Remove-Item (Join-Path $Model ".coli_usage") -Force -EA SilentlyContinue
  Get-Process qwen38 -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  if (Test-Path $MinGW) { $env:Path = "$MinGW;$env:Path" }
  FlushCache
  $c = RunExe $Colibri @("$Cap", "8", $promptFile) @{ SNAP=$Model; N_NEW="$Tokens"; OMP_NUM_THREADS="$Threads" }
  $c.Err -split "`r?`n" | Where-Object { $_ } | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
  $cText  = $c.Out.Trim()                       # colibri streams the generation to stdout
  Write-Host "  text> $cText" -ForegroundColor DarkGray
  $cLoad  = M $c.Err 'resident weights loaded in\s*([\d.]+)s'
  $cTTFT  = M $c.Err 'TTFT:\s*([\d.]+)\s*s'
  $cSpeed = M $c.Err 'Speed:\s*([\d.]+)\s*tok/s'
  $cHit   = M $c.Err 'hit rate:\s*([\d.]+)%'
  $cWall  = $c.Wall; $cPeak = $c.PeakGB
} else {
  Write-Host "`n=== colibri: not found at $Colibri - running qwen38-zig only ===" -ForegroundColor DarkYellow
}

# ============================ qwen38-zig ============================
Write-Host "`n=== qwen38-zig ===" -ForegroundColor Yellow
Get-Process qwen38-zig -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
FlushCache
$z = RunExe $Zig @("forward", $Model, "--tokens", $ids, "--steps", "$Tokens", "--expert-cap", "$Cap", "--threads", "$Threads", "--ram-limit", "28G") @{}
$z.Text -split "`r?`n" | Where-Object { $_ } | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
$zGreedy = ((M $z.Out 'greedy:\s*([\d, ]+)') -replace '\s','')
$zLoad   = M $z.Text 'model load:\s*([\d.]+)\s*s'
$zTTFT   = M $z.Text 'TTFT:\s*([\d.]+)\s*s'
$zSpeed  = M $z.Text 'decode:\s*([\d.]+)\s*tok/s'

# ============================= verdict =============================
$cCont = "-"; $match = "n/a (no colibri)"
if ($haveColibri) {
  $cIds = (& $Zig tokenize $Model --prompt ("$Prompt " + $cText)) 2>&1 | Out-String
  $cAll = (M $cIds 'ids \(\d+\): ([\d ]+)').Trim() -replace '\s+',','
  $cCont = (($cAll -split ',') | Select-Object -Skip $promptCount) -join ','
  $match = if ($cCont -eq $zGreedy) { "IDENTICAL" } else { "DIFFER" }
}

Write-Host "`n===================== RESULT =====================" -ForegroundColor Green
"{0,-16}{1,14}{2,14}" -f "",              "colibri",   "qwen38-zig"
"{0,-16}{1,14}{2,14}" -f "model load s",  $cLoad,      $zLoad
"{0,-16}{1,14}{2,14}" -f "TTFT s",        $cTTFT,      $zTTFT
"{0,-16}{1,14}{2,14}" -f "decode tok/s",  $cSpeed,     $zSpeed
"{0,-16}{1,14}{2,14}" -f "wall s",        $(if ($haveColibri) { [math]::Round($cWall,1) } else { "-" }), ([math]::Round($z.Wall,1))
"{0,-16}{1,14}{2,14}" -f "peak WS GB",    $cPeak,      $z.PeakGB
"{0,-16}{1,14}{2,14}" -f "expert hit %",  $cHit,       "-"
Write-Host "  (qwen38-zig WS includes reclaimable mmap page cache; its private" -ForegroundColor DarkGray
Write-Host "   allocation is about 7 GB - see 'tracked peak' from: qwen38-zig benchmark)" -ForegroundColor DarkGray
Write-Host ""
Write-Host ("output match : {0}" -f $match)
if ($haveColibri) { Write-Host ("  colibri    : $cCont") }
Write-Host ("  qwen38-zig : $zGreedy")
if ($haveColibri) { Write-Host ("  text       : `"$cText`"") }
