echo off
title Launch Helicoil Flask App
color 0A

:: Activate virtual environment
call C:\Users\ADMIN\Desktop\New_folder\.venv\Scripts\activate.bat

:: Start Flask app in new CMD window
start "Camera App Server" cmd /k python C:\Users\ADMIN\Desktop\New_folder\test_flask.py

:: Wait for the Flask server to start
timeout /t 7 >nul

:: Open default browser to localhost
start "" "http://localhost:5000"

echo Flask App is running at http://localhost:5000
echo Press any key to exit...
pause >/nul
