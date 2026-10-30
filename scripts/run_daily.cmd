@echo off
rem Task Scheduler entry: run daily MOLIT API sync, append output to logs\molit_daily.log
rem Exit code: 0 ok (incl. daily cap stop), 1 some jobs failed (retried next day), 2 key/permission error
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONUTF8=1
set UV=uv
where uv > nul 2>&1 || set UV="%USERPROFILE%\.local\bin\uv.exe"
%UV% run python scripts\daily_molit_api.py %* >> logs\molit_daily.log 2>&1
exit /b %ERRORLEVEL%
