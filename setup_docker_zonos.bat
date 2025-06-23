@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET "SCRIPT_EXIT_CODE=0"

REM === Configuration ===
SET "SCRIPT_NAME=setup_docker_zonos.bat"
SET "ZONOS_REPO_URL=https://github.com/Zyphra/Zonos.git"
SET "ZONOS_SRC_DIR=Zonos_src"
SET "WUBU_ZONOS_IMAGE_TAG=wubu_zonos_image"

ECHO [!SCRIPT_NAME!] Starting Zonos Docker Setup...

REM === Main Logic ===
CALL :CheckDocker
REM Note: :CheckDocker behavior is modified.
REM If Docker seems unresponsive, it will warn but script will proceed to attempt build.

IF !ERRORLEVEL! NEQ 0 (
    REM This check is now for critical failures within :CheckDocker itself (e.g. docker command not found)
    REM OR if :BuildZonosImageFromSource (called from :CheckDocker) fails.
    ECHO [!SCRIPT_NAME!] A critical error occurred during Docker setup or Zonos image build.
    SET "SCRIPT_EXIT_CODE=!ERRORLEVEL!"
    GOTO :HandleExit
)

REM === Completion (if all successful) ===
ECHO.
ECHO =====================================================================
ECHO [!SCRIPT_NAME!] Zonos Docker Setup completed.
IF DEFINED DOCKER_SETUP_SUCCESS (
    IF "!DOCKER_SETUP_SUCCESS!"=="true" (
        ECHO [!SCRIPT_NAME!] Zonos Docker image "!WUBU_ZONOS_IMAGE_TAG!" should be ready.
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Zonos Docker image build may have encountered issues (check logs).
    )
) ELSE (
    ECHO [!SCRIPT_NAME!] Zonos Docker image build status unknown (check logs).
)
ECHO =====================================================================
SET "SCRIPT_EXIT_CODE=0" REM Explicitly set for success path
GOTO :HandleExit


REM === Subroutines ===
:CheckDocker
    ECHO.
    ECHO [!SCRIPT_NAME!] Checking for Docker (required for Zonos TTS)...
    SET "DOCKER_SETUP_SUCCESS=false" REM Flag to track overall Docker setup for Zonos

    docker --version >NUL 2>NUL
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] WARNING: Docker command not found on system PATH.
        CALL :DockerInstallGuide
        ECHO [!SCRIPT_NAME!] Please re-run this script after installing Docker and ensuring it's in PATH.
        ECHO [!SCRIPT_NAME!] Attempting Zonos build anyway, but it is likely to fail.
        REM Do not EXIT /B here, proceed to build attempt
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Docker command is accessible. Checking if Docker engine is responsive...
        docker version >NUL 2>NUL
        IF !ERRORLEVEL! NEQ 0 (
            ECHO [!SCRIPT_NAME!] WARNING: Docker command found, but Docker engine may not be responsive or `docker version` failed (Error Code: !ERRORLEVEL!).
            ECHO [!SCRIPT_NAME!] This script will attempt to build the Zonos image anyway.
            ECHO [!SCRIPT_NAME!] If the build fails, please ensure Docker Desktop is running correctly and has finished initializing.
            CALL :DockerInstallGuide
            REM Do not EXIT /B here, proceed to build attempt
        ) ELSE (
            ECHO [!SCRIPT_NAME!] Docker engine appears responsive.
        )
    )

    CALL :BuildZonosImageFromSource
    IF !ERRORLEVEL! EQU 0 (
        SET "DOCKER_SETUP_SUCCESS=true"
    ) ELSE (
        ECHO [!SCRIPT_NAME!] ERROR: The Zonos Docker image preparation process failed. Zonos TTS may not function. (Reported Error Code from build: !ERRORLEVEL!)
        REM Propagate the error level from BuildZonosImageFromSource
        EXIT /B !ERRORLEVEL!
    )
EXIT /B 0

:DockerInstallGuide
    ECHO.
    ECHO ================================================================================
    ECHO [!SCRIPT_NAME!] IMPORTANT: Docker Desktop Installation Guide (for Zonos TTS)
    ECHO.
    ECHO [!SCRIPT_NAME!] WuBu's Zonos Text-to-Speech engine requires a Docker container.
    ECHO.
    ECHO [!SCRIPT_NAME!] 1. Download Docker Desktop for Windows from the official website:
    ECHO [!SCRIPT_NAME!]    https://www.docker.com/products/docker-desktop/
    ECHO.
    ECHO [!SCRIPT_NAME!] 2. Run the installer and follow the instructions. This might require a system restart.
    ECHO.
    ECHO [!SCRIPT_NAME!] 3. After installation, ensure Docker Desktop is RUNNING.
    ECHO [!SCRIPT_NAME!]    (Look for the Docker whale icon in your system tray and ensure it's not indicating an error).
    ECHO [!SCRIPT_NAME!]    It might take a few minutes for Docker Desktop to fully initialize after starting.
    ECHO.
    ECHO [!SCRIPT_NAME!] 4. IMPORTANT: You may need to restart your Command Prompt or PowerShell
    ECHO [!SCRIPT_NAME!]    (and this script) for the 'docker' command to be recognized and for it to
    ECHO [!SCRIPT_NAME!]    connect to the Docker engine properly.
    ECHO [!SCRIPT_NAME!]    Test by opening a NEW terminal and typing: docker --version
    ECHO [!SCRIPT_NAME!]    If it still fails, Docker Desktop might not be running correctly.
    ECHO ================================================================================
    ECHO.
    PAUSE "[!SCRIPT_NAME!] Press any key to continue after attempting Docker Desktop installation and ensuring it's running..."
EXIT /B 0

:BuildZonosImageFromSource
    ECHO.
    ECHO [!SCRIPT_NAME!] Preparing to build Zonos Docker image (!WUBU_ZONOS_IMAGE_TAG!) from source.
    ECHO [!SCRIPT_NAME!] This involves cloning the Zonos repository (!ZONOS_REPO_URL!) and running 'docker build'.
    ECHO [!SCRIPT_NAME!] This may take a significant amount of time and disk space.

    REM Clone Zonos repository if Zonos_src directory doesn't exist
    IF NOT EXIST "%~dp0!ZONOS_SRC_DIR!" (
        ECHO [!SCRIPT_NAME!] Zonos source directory "!ZONOS_SRC_DIR!" not found. Cloning from !ZONOS_REPO_URL!...
        git clone "!ZONOS_REPO_URL!" "!ZONOS_SRC_DIR!"
        SET "GIT_CLONE_ERRORLEVEL=!ERRORLEVEL!"
        IF !GIT_CLONE_ERRORLEVEL! NEQ 0 (
            ECHO [!SCRIPT_NAME!] ERROR: Failed to clone Zonos repository from !ZONOS_REPO_URL!. (Error Code: !GIT_CLONE_ERRORLEVEL!)
            ECHO [!SCRIPT_NAME!] Please check your internet connection, Git installation, and repository URL. Zonos TTS cannot be set up.
            EXIT /B 1
        )
        ECHO [!SCRIPT_NAME!] Zonos repository cloned successfully into "!ZONOS_SRC_DIR!".
    ) ELSE (
        ECHO [!SCRIPT_NAME!] Zonos source directory "!ZONOS_SRC_DIR!" already exists. Using existing directory.
        ECHO [!SCRIPT_NAME!] To ensure you build with the latest Zonos source, you may want to manually delete "!ZONOS_SRC_DIR!" and re-run this script.
    )

    REM Verify Dockerfile exists in the Zonos source directory before attempting to build
    IF NOT EXIST "%~dp0!ZONOS_SRC_DIR!\Dockerfile" (
        ECHO [!SCRIPT_NAME!] ERROR: 'Dockerfile' is missing in the Zonos source directory "%~dp0!ZONOS_SRC_DIR!".
        ECHO [!SCRIPT_NAME!] Cannot build the Zonos Docker image. The Zonos repository might be corrupted or incomplete.
        ECHO [!SCRIPT_NAME!] Try deleting the "%~dp0!ZONOS_SRC_DIR!" directory and re-running this script.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Building Zonos Docker image (!WUBU_ZONOS_IMAGE_TAG!) from source in "!ZONOS_SRC_DIR!"...
    ECHO [!SCRIPT_NAME!] This will use the Dockerfile provided within the Zonos repository.
    PUSHD "%~dp0!ZONOS_SRC_DIR!"
    IF !ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Could not change directory to "%~dp0!ZONOS_SRC_DIR!". (Error Code: !ERRORLEVEL!)
        EXIT /B 1
    )

    docker build -t "!WUBU_ZONOS_IMAGE_TAG!" .
    SET "DOCKER_BUILD_ERRORLEVEL=!ERRORLEVEL!"
    POPD
    IF !DOCKER_BUILD_ERRORLEVEL! NEQ 0 (
        ECHO [!SCRIPT_NAME!] ERROR: Failed to build Zonos Docker image "!WUBU_ZONOS_IMAGE_TAG!" from source. (Error Code: !DOCKER_BUILD_ERRORLEVEL!)
        ECHO [!SCRIPT_NAME!] Please check Docker Desktop is running correctly, and review any errors from the 'docker build' command output above.
        EXIT /B 1
    )

    ECHO [!SCRIPT_NAME!] Successfully built Zonos Docker image: !WUBU_ZONOS_IMAGE_TAG!.
EXIT /B 0

REM === Final Exit Point ===
:HandleExit
ECHO.
IF "!SCRIPT_EXIT_CODE!" NEQ "0" (
    ECHO [!SCRIPT_NAME!] Script finished with errors. Error Code: !SCRIPT_EXIT_CODE!
) ELSE (
    ECHO [!SCRIPT_NAME!] Script finished successfully.
)
ENDLOCAL & (
    REM Removed PAUSE from sub-script, master script can decide to PAUSE
    EXIT /B %SCRIPT_EXIT_CODE%
)
