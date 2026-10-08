@echo off
setlocal
cd /d "%~dp0"
python documentation\update.py %*
exit /b %errorlevel%
