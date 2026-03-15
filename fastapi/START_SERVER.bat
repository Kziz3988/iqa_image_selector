if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
pause