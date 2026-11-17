@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0scripts\start.ps1" -OpenBrowser %*
set "launchExit=%ERRORLEVEL%"
if not "%launchExit%"=="0" pause
exit /b %launchExit%
