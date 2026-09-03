<#
  bench_ab2.ps1 - cold + warm A/B: colibri vs qwen38-zig, same checkpoint / prompt
  / token count / cap / threads / greedy.

  For each engine:
    1. flush OS page cache (RAM ballast), wipe learned priors
    2. run once            -> COLD  (empty page cache, no priors)
    3. run again, no flush  -> WARM  (hot page cache + priors written by run 2)

  Phase 9b (MoE per-expert threading) only helps once the expert cache is hot, so
  the WARM row is the "parity on the same hardware" number. COLD is the honest
  start-from-nothing number.
#>
param(
  # Path to the Qwen3.8-Flash-Next-FP8 checkpoint dir (or set $env:QWEN38_MODEL).
  [string]$Model   = $(if ($env:QWEN38_MODEL) { $env:QWEN38_MODEL } else { "D:\Models\Qwen38-FP8" }),
  [string]$Prompt  = "The capital of France is",
  [int]   $Tokens  = 24,
  [int]   $Cap     = 64,
  [int]   $Threads = [Environment]::ProcessorCount,
  [string]$Zig     = (Join-Path $PSScriptRoot "zig-out\bin\qwen38-zig.exe"),
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
  $o = Join-Path $env:TEMP "ab2_out.txt"; $e = Join-Path $env:TEMP "ab2_err.txt"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru `
        -RedirectStandardOutput $o -RedirectStandardError $e
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

$tk = (& $Zig tokenize $Model --prompt $Prompt) 2>&1 | Out-String
$ids = (M $tk 'ids \(\d+\): ([\d ]+)').Trim() -replace '\s+', ','
if ($ids -eq '?' -or -not $ids) { throw "could not tokenize the prompt" }
$promptCount = ($ids -split ',').Count
Write-Host ""
Write-Host "prompt : `"$Prompt`"  ->  ids: $ids" -ForegroundColor Cyan
Write-Host "config : $Tokens new tokens - cap $Cap/layer - $Threads threads - greedy" -ForegroundColor Cyan

$promptFile = Join-Path $env:TEMP "ab2_prompt.txt"
[System.IO.File]::WriteAllText($promptFile, $Prompt)

$haveColibri = Test-Path $Colibri

# ============================== colibri ==============================
if ($haveColibri) {
  Write-Host "`n=== colibri (cold, then warm) ===" -ForegroundColor Yellow
  Remove-Item (Join-Path $Model ".coli_usage") -Force -EA SilentlyContinue
  Get-Process qwen38 -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  if (Test-Path $MinGW) { $env:Path = "$MinGW;$env:Path" }
  FlushCache
  $cCold = RunExe $Colibri @("$Cap", "8", $promptFile) @{ SNAP=$Model; N_NEW="$Tokens"; OMP_NUM_THREADS="$Threads" }
  Write-Host "  cold text> $($cCold.Out.Trim())" -ForegroundColor DarkGray
  $cWarm = RunExe $Colibri @("$Cap", "8", $promptFile) @{ SNAP=$Model; N_NEW="$Tokens"; OMP_NUM_THREADS="$Threads" }
  Write-Host "  warm text> $($cWarm.Out.Trim())" -ForegroundColor DarkGray
} else {
  Write-Host "`n=== colibri: not found at $Colibri - skipping the A/B half ===" -ForegroundColor DarkYellow
}

# ============================ qwen38-zig ============================
Write-Host "`n=== qwen38-zig (cold, then warm) ===" -ForegroundColor Yellow
Remove-Item (Join-Path $Model ".colizig_usage") -Force -EA SilentlyContinue
Get-Process qwen38-zig -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
FlushCache
$zArgs = @("forward", $Model, "--tokens", $ids, "--steps", "$Tokens", "--expert-cap", "$Cap", "--threads", "$Threads", "--ram-limit", "28G")
$zCold = RunExe $Zig $zArgs @{}
$zWarm = RunExe $Zig $zArgs @{}

# ============================= verdict =============================
function ColiCont($text) {
  $r = (& $Zig tokenize $Model --prompt ("$Prompt " + $text.Trim())) 2>&1 | Out-String
  $all = (M $r 'ids \(\d+\): ([\d ]+)').Trim() -replace '\s+',','
  (($all -split ',') | Select-Object -Skip $promptCount) -join ','
}
$zGreedyCold = ((M $zCold.Out 'greedy:\s*([\d, ]+)') -replace '\s','')
$zGreedyWarm = ((M $zWarm.Out 'greedy:\s*([\d, ]+)') -replace '\s','')
$cc = if ($haveColibri) { M $cCold.Err 'resident weights loaded in\s*([\d.]+)s' } else { "-" }

function Row($label, $a, $b) { "{0,-16}{1,16}{2,16}" -f $label, $a, $b }
function CM($obj, $rx) { if ($haveColibri) { M $obj $rx } else { "-" } }

Write-Host "`n===================== COLD =====================" -ForegroundColor Green
Row ""              "colibri" "qwen38-zig"
Row "model load s"  $cc                                                     (M $zCold.Text 'model load:\s*([\d.]+)\s*s')
Row "TTFT s"        (CM $cCold.Err 'TTFT:\s*([\d.]+)\s*s')                   (M $zCold.Text 'TTFT:\s*([\d.]+)\s*s')
Row "decode tok/s"  (CM $cCold.Err 'Speed:\s*([\d.]+)\s*tok/s')             (M $zCold.Text 'decode:\s*([\d.]+)\s*tok/s')
Row "wall s"        $(if ($haveColibri) { [math]::Round($cCold.Wall,1) } else { "-" })  ([math]::Round($zCold.Wall,1))
Row "peak WS GB"    $(if ($haveColibri) { $cCold.PeakGB } else { "-" })     $zCold.PeakGB

Write-Host "`n===================== WARM =====================" -ForegroundColor Green
Row ""              "colibri" "qwen38-zig"
Row "TTFT s"        (CM $cWarm.Err 'TTFT:\s*([\d.]+)\s*s')                   (M $zWarm.Text 'TTFT:\s*([\d.]+)\s*s')
Row "decode tok/s"  (CM $cWarm.Err 'Speed:\s*([\d.]+)\s*tok/s')             (M $zWarm.Text 'decode:\s*([\d.]+)\s*tok/s')
Row "wall s"        $(if ($haveColibri) { [math]::Round($cWarm.Wall,1) } else { "-" })  ([math]::Round($zWarm.Wall,1))
Row "peak WS GB"    $(if ($haveColibri) { $cWarm.PeakGB } else { "-" })     $zWarm.PeakGB
Row "colibri hit %" (CM $cWarm.Err 'hit rate:\s*([\d.]+)%')                  "n/a"
Row "zig demand ld" "n/a"                                                   (M $zWarm.Text 'expert demand loads:\s*(\d+)')

Write-Host "`n----------------- output -----------------" -ForegroundColor Green
if ($haveColibri) {
  $cContWarm = ColiCont $cWarm.Out
  Write-Host ("cold match : {0}" -f $(if ((ColiCont $cCold.Out) -eq $zGreedyCold) { "IDENTICAL" } else { "DIFFER" }))
  Write-Host ("warm match : {0}" -f $(if ($cContWarm -eq $zGreedyWarm) { "IDENTICAL" } else { "DIFFER" }))
  Write-Host ("  colibri    : $cContWarm")
}
Write-Host ("  qwen38-zig : $zGreedyWarm")
Write-Host ("`ndone.") -ForegroundColor Green
