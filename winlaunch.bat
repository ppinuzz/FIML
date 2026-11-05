:: WINDOWS BATCH FILE TO SET UP MINIFORGE

:: echo off		disables command echoing starting from the NEXT command
:: @ 			prevent current command from being printed
@echo off
set "ENV_NAME=FIML"
set "RELATIVE_WORK_DIR=examples\Burgers"
:: call mamba's activation script (used to initialise several things for the prompt)
:: %USERPROFILE% contains usually C:\Users\<username>
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" "%ENV_NAME%"
:: /d	change also current drive (on Windows the path doesn't contain the drive??)
:: %0	path of this script
:: ~	remove surrounding "..."
:: d	extract drive letter (here, C:)
:: p 	extract directory path
:: concatenate the script parent directory to the relative work directory
:: e.g. if the full path is C:\PhD\myCode, then
::	%0		C:\PhD\myCode\winlaunch.bat
::	%~d0	C:
::	%~p0	\PhD\myCode\
::	%~dp0	C:\PhD\myCode\
::cd /d "%~dp0"
cd /d "%~dp0%RELATIVE_WORK_DIR%"
:: open Windows command prompt and keep it open (/k)
:: (Miniforge prompt is just Windows cmd called after executing the activation scrip)
cmd /k