@echo off
set MINERU_MODEL_SOURCE=modelscope
set TOKENIZERS_PARALLELISM=false

echo ==========================================
echo   Project2 Pipeline
echo ==========================================
echo   Start: %DATE% %TIME%
echo ==========================================

echo.
echo [1/4] MinerU Parse ...
python offline\parse_pdf_mineru.py 招股说明书2
if errorlevel 1 (
    echo FAIL MinerU
    pause
    exit /b
)
echo OK

echo.
echo [2/4] Chunk ...
python offline\chunk_text.py
if errorlevel 1 (
    echo FAIL Chunk
    pause
    exit /b
)
echo OK

echo.
echo [3/4] Embeddings ...
python offline\generate_embeddings.py 招股说明书2
if errorlevel 1 (
    echo FAIL Embed
    pause
    exit /b
)
echo OK

echo.
echo [4/4] Milvus Import ...
python offline\import_to_milvus.py
if errorlevel 1 (
    echo FAIL Milvus
    pause
    exit /b
)
echo OK

echo.
echo ==========================================
echo   Done!
echo ==========================================
pause
