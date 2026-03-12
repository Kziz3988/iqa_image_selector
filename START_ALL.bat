@echo off
taskkill /F /IM node.exe /T
taskkill /F /IM python.exe /T
start "" cmd /c "cd vue && START_SERVER.bat"
start "" cmd /c "cd fastapi && START_SERVER.bat"