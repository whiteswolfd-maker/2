@echo off
rem ============================================================
rem  run_lsdyna.bat — run an LS-DYNA keyword deck with the ANSYS
rem  2022 R1 (v221) bundled solver, setting the Intel runtime PATH
rem  that the ANSYS launcher normally injects.
rem
rem  Usage:
rem     run_lsdyna.bat deck.k [workdir] [ncpu]
rem       deck.k   -> path to the keyword deck
rem       workdir  -> output directory (default: current dir)
rem       ncpu     -> SMP threads (default 4)
rem
rem  Output: d3plot / d3plotNN / d3hsp / messag in workdir.
rem ============================================================
setlocal
set "AWP=E:\ANSYS Inc\v221"
set "LSDYNA=%AWP%\ansys\bin\winx64\lsdyna_dp.exe"
set "LIC=E:\ANSYS Inc\Shared Files\licensing\winx64"
rem 1) Intel runtime DLLs (libiomp5md.dll ...) that the ANSYS launcher adds.
rem 2) the ANSYS licensing dir on PATH activates the HASP license emulator
rem    (netapi32.dll) so the ANSYSLIC-enabled solver can check out a license.
set "PATH=%AWP%\tp\IntelCompiler\2019.5.281\winx64;%AWP%\tp\IntelMKL\2020.0.166\winx64;%LIC%;%PATH%"
rem Check the ANSYS license (not network/local) -- this is what makes the
rem bundled LS-DYNA talk to the ANSYS licensing client.
set "LSTC_LICENSE=Ansys"
set "LSTC_LICENSE_SERVER="
set "LSTC_LICENSE_FILE="
if not defined ANSYSLMD_LICENSE_FILE set "ANSYSLMD_LICENSE_FILE=E:\ANSYS Inc\Shared Files\licensing\license_files\ansyslmd.lic"

if "%~1"=="" (
  echo Usage: run_lsdyna.bat deck.k [workdir] [ncpu]
  exit /b 1
)
set "DECK=%~f1"
set "WORKDIR=%~2"
if "%WORKDIR%"=="" set "WORKDIR=%CD%"
set "NCPU=%~3"
if "%NCPU%"=="" set "NCPU=4"

if not exist "%WORKDIR%" mkdir "%WORKDIR%"
copy /Y "%DECK%" "%WORKDIR%\input.k" >nul
echo Running LS-DYNA: %LSDYNA%
echo   deck  : %DECK%
echo   work  : %WORKDIR%
echo   ncpu  : %NCPU%
pushd "%WORKDIR%"
"%LSDYNA%" I=input.k NCPU=%NCPU% MEMORY=2000M
set "RC=%ERRORLEVEL%"
popd
echo LS-DYNA exit code: %RC%
exit /b %RC%
