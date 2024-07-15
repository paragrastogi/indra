@ECHO off

set /P installer=Enter the name of your preferred package manager (in lower case):

if "%installer%" == "conda" (
conda install --file py_packages.txt
) else (
pip install -r py_packages.txt
)

ECHO "Finished!"
EXIT 