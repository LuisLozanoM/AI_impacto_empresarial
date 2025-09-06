@echo off
echo Starting AI Language Translator...
echo.
cd /d "C:\Users\L03055876\Desktop\AI_impacto_empresarial"
call ai_ml_env\Scripts\activate.bat
streamlit run translation_app.py
pause
