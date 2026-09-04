<#  build_cuda.ps1 - build the optional CUDA backend (colizig_cuda.dll).

    `zig build cuda` needs nvcc AND an MSVC host compiler (cl.exe) on PATH.
    This wrapper imports the Visual Studio "x64 Native Tools" environment, puts
    the CUDA toolkit bin on PATH, then runs `zig build cuda`. The DLL lands in
    zig-out\bin\ next to colizig.exe; the engine loads it at runtime with --cuda.

    Override paths with -Vcvars / -CudaBin / -Zig if they differ on your box.
#>
param(
  [string]$Vcvars  = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin" }),
  [string]$Zig     = "zig",
  [string]$Arch    = "native"
)

if (-not (Test-Path $Vcvars))  { throw "vcvars64.bat not found at $Vcvars - install the VS Build Tools C++ workload, or pass -Vcvars" }
if (-not (Test-Path $CudaBin)) { throw "CUDA bin not found at $CudaBin - pass -CudaBin" }

Write-Host "importing MSVC environment ..." -ForegroundColor DarkGray
cmd /c "`"$Vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:$($matches[1])" $matches[2] }
}
$env:Path = "$CudaBin;$env:Path"

if (-not (Get-Command cl.exe   -EA SilentlyContinue)) { throw "cl.exe still not on PATH after vcvars" }
if (-not (Get-Command nvcc     -EA SilentlyContinue)) { throw "nvcc not on PATH" }
Write-Host ("cl   : " + (Get-Command cl.exe).Source)   -ForegroundColor DarkGray
Write-Host ("nvcc : " + (Get-Command nvcc).Source)     -ForegroundColor DarkGray

& $Zig build cuda "-Dcuda-arch=$Arch" --verbose
if ($LASTEXITCODE -ne 0) { throw "zig build cuda failed ($LASTEXITCODE)" }
Write-Host "`nbuilt: zig-out\bin\colizig_cuda.dll" -ForegroundColor Green
Write-Host "run:   .\zig-out\bin\colizig.exe chat <MODEL_DIR> --cuda" -ForegroundColor Green
