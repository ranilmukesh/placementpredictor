@echo off
title PlacementPredictor+ Launcher
color 0A

echo ============================================================
echo          PlacementPredictor+ - AI Career Router
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo [1/6] Installing/Updating dependencies...
pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn joblib imbalanced-learn pydantic agno sqlalchemy python-dotenv --quiet
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install.
) else (
    echo       Done!
)
echo.

:: Also clean legacy CardioSense artifacts if present
if exist "cardiosense_artifacts.pkl" (
    del /f /q "cardiosense_artifacts.pkl"
    echo       Legacy CardioSense artifacts removed.
)

:: Smart PKL Caching: check if --force-train was passed
set FORCE_TRAIN=0
for %%a in (%*) do (
    if "%%a"=="--force-train" set FORCE_TRAIN=1
)

if "%FORCE_TRAIN%"=="1" (
    echo [2/6] Force-train requested. Deleting old artifacts...
    if exist "placement_artifacts.pkl" del /f /q "placement_artifacts.pkl"
    echo       Old artifacts deleted!
    goto :DO_TRAIN
)

:: If pkl exists, skip training
if exist "placement_artifacts.pkl" (
    echo [2/6] Found existing model artifacts. Skipping training...
    echo       TIP: Use --force-train flag to retrain: start_placement_predictor.bat --force-train
    goto :SKIP_TRAIN
)

echo [2/6] No model artifacts found. Training required...

:DO_TRAIN
:: Set Nvidia API Key first (needed before training)
echo [3/6] Setting up AI Chat credentials...
setx NVIDIA_API_KEY "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" >nul 2>&1
set NVIDIA_API_KEY=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
echo NVIDIA_API_KEY=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> .env
echo       NVIDIA_API_KEY configured!
echo.

echo [4/6] Training PlacementPredictor+ model...
echo       This trains the XGBoost Classifier for Placement.
echo.
python train_model.py
if errorlevel 1 (
    echo [ERROR] Model training failed!
    echo Please check if the placement CSV dataset files exist in this folder.
    pause
    exit /b 1
)
echo       Model trained successfully!
echo.
goto :START_SERVERS

:SKIP_TRAIN
:: Set Nvidia API Key
echo [3/6] Setting up AI Chat credentials...
setx NVIDIA_API_KEY "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" >nul 2>&1
set NVIDIA_API_KEY=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
echo NVIDIA_API_KEY=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> .env
echo       NVIDIA_API_KEY configured!
echo.
echo [4/6] Skipped training (using cached model).
echo.

:START_SERVERS
echo [5/6] Starting API Server on http://127.0.0.1:7860 ...
:: Fixed: Host 0.0.0.0 and Port 7860 to match main.py and HF patterns
start "PlacementPredictor+ API" cmd /k "title PlacementPredictor+ API Server && color 0B && set NVIDIA_API_KEY=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 && python main.py"
echo       Waiting for API to start...
timeout /t 8 /nobreak >nul

:: Validate the model loaded correctly by hitting /health
echo       Validating model artifacts...
curl -s http://127.0.0.1:7860/health 2>nul | findstr /C:"model_loaded" | findstr /C:"true" >nul 2>&1
if errorlevel 1 (
    echo [!] Model failed to load - version mismatch or corrupted pkl.
    echo       Killing API server and retraining...
    taskkill /FI "WINDOWTITLE eq PlacementPredictor+ API Server" /F >nul 2>&1
    timeout /t 2 /nobreak >nul
    if exist "placement_artifacts.pkl" del /f /q "placement_artifacts.pkl"
    echo.
    echo [4/6] Retraining model...
    python train_model.py
    if errorlevel 1 (
        echo [ERROR] Retraining also failed!
        pause
        exit /b 1
    )
    echo       Retrained successfully!
    echo.
    echo [5/6] Restarting API Server...
    start "PlacementPredictor+ API" cmd /k "title PlacementPredictor+ API Server && color 0B && set NVIDIA_API_KEY=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 && python main.py"
    timeout /t 8 /nobreak >nul
)
echo       API Server started!
echo.

echo ============================================================
echo                    SERVERS ARE RUNNING!
echo ============================================================
echo.
echo   Application UI:  http://127.0.0.1:7860
echo.
echo   TIP: Use the "Demo Data" button in the UI for quick testing
echo   TIP: Click the chat bubble after results for AI assistant
echo   TIP: Use --force-train flag to force model retraining
echo.
echo ============================================================
echo.

:: Open the frontend in default browser
echo Opening PlacementPredictor+ in your browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:7860

echo.
echo Press any key to close this launcher (servers will keep running)...
pause >nul
