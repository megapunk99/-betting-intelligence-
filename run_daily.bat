@echo off
REM Daily NBA Pipeline — Launched by Windows Task Scheduler
cd /D "%~dp0"
C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe -B tools/daily_run.py
