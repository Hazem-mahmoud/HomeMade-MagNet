@echo off
echo Cleaning old documentation...
if exist html rmdir /s /q html
if exist _doxygen rmdir /s /q _doxygen

echo Generating new Doxygen documentation...
"C:\Program Files\doxygen\bin\doxygen.exe" Doxyfile

echo Done! The documentation is in the 'html' folder.
pause
