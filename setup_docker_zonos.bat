@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_OVERALL_EXIT_CODE=0" 

REM === Configuration ===
SET "SCRIPT_NAME=setup_docker_zonos.bat"
SET "ZONOS_REPO_URL=https://github.com/Zyphra/Zonos.git"
SET "ZONOS_SRC_DIR=Zonos_src"
SET "WUBU_ZONOS_IMAGE_TAG=wubu_zonos_image"

ECHO [!SCRIPT_NAME!] Starting Zonos Docker Setup...

REM === Docker Prerequisite - User Prompt ===
ECHO.
ECHO ==== Docker Prerequisite - User Prompt ====
PAUSE "[!SCRIPT_NAME!] Press any key to continue ONCE DOCKER DESKTOP IS RUNNING..."
ECHO.
ECHO [!SCRIPT_NAME!] Proceeding with Zonos image build attempt...

REM === Main Logic - Build Image ===
CALL :BuildZonosImageAndReport
SET "SCRIPT_OVERALL_EXIT_CODE=!ERRORLEVEL!"
ECHO [DEBUG MainLogic] SCRIPT_OVERALL_EXIT_CODE after call is: '!SCRIPT_OVERALL_EXIT_CODE!'

ECHO.
ECHO ==== Build Result Summary ====
ECHO [DEBUG MainLogic] About to execute: IF "!SCRIPT_OVERALL_EXIT_CODE!" EQU "0" (
IF "!SCRIPT_OVERALL_EXIT_CODE!" EQU "0" (
    ECHO [!SCRIPT_NAME!] Docker build command reported SUCCESS (Exit Code: 0^).
    ECHO [!SCRIPT_NAME!] Zonos Docker image "!WUBU_ZONOS_IMAGE_TAG!" should be ready.
    ECHO [!SCRIPT_NAME!] Use 'docker images' to verify.
    ECHO ==== END SUCCESS MSG ====
    PAUSE "[!SCRIPT_NAME!] SUCCESS. Review output. Press key to prepare for exit..."
) ELSE (
    ECHO [!SCRIPT_NAME!] Docker build command reported FAILURE (Exit Code: !SCRIPT_OVERALL_EXIT_CODE!^).
    ECHO [!SCRIPT_NAME!] Please review the Docker build output above for errors.
    ECHO ==== END FAILURE MSG ====
    PAUSE "[!SCRIPT_NAME!] FAILURE. Review output. Press key to prepare for exit..."
)
ECHO [DEBUG MainLogic] Reached after IF/ELSE block.

REM === Final Exit Point ===
:HandleExitNoMatterWhat
    ECHO [DEBUG FinalExit] In HandleExitNoMatterWhat. SCRIPT_OVERALL_EXIT_CODE is !SCRIPT_OVERALL_EXIT_CODE!
    SET FINAL_CODE_TO_EXIT_WITH=!SCRIPT_OVERALL_EXIT_CODE!
    ENDLOCAL
    EXIT /B %FINAL_CODE_TO_EXIT_WITH%


REM === Subroutines ===
:BuildZonosImageAndReport
    ECHO.
    ECHO [!SCRIPT_NAME!] In :BuildZonosImageAndReport
    SET "FULL_ZONOS_SRC_DIR=%~dp0!ZONOS_SRC_DIR!"

    IF NOT EXIST "!FULL_ZONOS_SRC_DIR!" (
        ECHO [!SCRIPT_NAME!] Src dir not found. Cloning...
        git clone "!ZONOS_REPO_URL!" "!FULL_ZONOS_SRC_DIR!"
        IF !ERRORLEVEL! NEQ 0 (EXIT /B 11)
        ECHO [!SCRIPT_NAME!] Cloned.
        IF NOT EXIST "!FULL_ZONOS_SRC_DIR!" (EXIT /B 111)
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Src dir exists.
    )

    IF NOT EXIST "!FULL_ZONOS_SRC_DIR!\Dockerfile" (
        ECHO [!SCRIPT_NAME!] ERROR: Dockerfile missing.
        EXIT /B 12 
    )
    
    ECHO [!SCRIPT_NAME!] Building image !WUBU_ZONOS_IMAGE_TAG!...
    
    docker.exe build -t "!WUBU_ZONOS_IMAGE_TAG!" "!FULL_ZONOS_SRC_DIR!"
    SET "DOCKER_BUILD_RAW_EXIT_CODE=!ERRORLEVEL!"
    ECHO [DEBUG Sub] Raw Exit Code from 'docker build': !DOCKER_BUILD_RAW_EXIT_CODE!
    
EXIT /B !DOCKER_BUILD_RAW_EXIT_CODE!