@echo off
setlocal
REM 检查虚拟环境是否存在 pip
if not exist ".venv\Scripts\pip.exe" (
    echo pip not found in "%VENV_PATH%\Scripts"
    exit /b 1
)
REM 生成 requirements.txt
echo Generating requirements.txt...
".venv\Scripts\pip.exe" freeze > requirements.txt
echo requirements.txt generated successfully.
pause