import psutil # type: ignore
import logging
from typing import Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

def get_cpu_usage(interval: float = 0.5) -> Dict[str, Union[str, float]]:
    """
    Gets the current system-wide CPU utilization percentage.

    Args:
        interval: The interval in seconds over which to measure CPU usage.
                  A small interval is needed for a non-blocking call to compare
                  CPU times before and after.

    Returns:
        A dictionary with "cpu_percent" or "error".
    """
    try:
        # psutil.cpu_percent with interval is blocking but more accurate for system-wide usage.
        cpu_percent = psutil.cpu_percent(interval=interval)
        logger.info(f"Current CPU usage: {cpu_percent}%")
        return {"cpu_percent": cpu_percent}
    except Exception as e:
        logger.error(f"Error getting CPU usage: {e}", exc_info=True)
        return {"error": f"Could not retrieve CPU usage: {e}"}

def get_memory_usage() -> Dict[str, Any]:
    """
    Gets the current system memory usage statistics.

    Returns:
        A dictionary with "total_gb", "available_gb", "percent_used", "used_gb" or "error".
    """
    try:
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024**3), 2)
        available_gb = round(mem.available / (1024**3), 2)
        used_gb = round(mem.used / (1024**3), 2)
        percent_used = mem.percent

        logger.info(f"Memory Usage: {used_gb}GB / {total_gb}GB ({percent_used}%) used. Available: {available_gb}GB")
        return {
            "total_gb": total_gb,
            "available_gb": available_gb,
            "used_gb": used_gb,
            "percent_used": percent_used
        }
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}", exc_info=True)
        return {"error": f"Could not retrieve memory usage: {e}"}

def get_disk_usage(path: str = "/") -> Dict[str, Any]:
    """
    Gets disk usage statistics for the partition that contains the given path.

    Args:
        path: The path to check the disk usage for (e.g., "/", "C:\\").
              Defaults to the root directory.

    Returns:
        A dictionary with "path", "total_gb", "used_gb", "free_gb", "percent_used" or "error".
    """
    try:
        usage = psutil.disk_usage(path)
        total_gb = round(usage.total / (1024**3), 2)
        used_gb = round(usage.used / (1024**3), 2)
        free_gb = round(usage.free / (1024**3), 2)
        percent_used = usage.percent

        logger.info(f"Disk Usage for '{path}': Used {used_gb}GB / {total_gb}GB ({percent_used}%). Free: {free_gb}GB")
        return {
            "path": path,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "percent_used": percent_used
        }
    except FileNotFoundError:
        logger.error(f"Disk path not found: {path}")
        return {"error": f"Disk path not found: {path}"}
    except Exception as e:
        logger.error(f"Error getting disk usage for '{path}': {e}", exc_info=True)
        return {"error": f"Could not retrieve disk usage for '{path}': {e}"}

def get_battery_status() -> Dict[str, Any]:
    """
    Gets the current battery status (if available).

    Returns:
        A dictionary with "percent", "secsleft", "power_plugged" or "error" / "not_available".
        "secsleft" can be psutil.POWER_TIME_UNLIMITED or psutil.POWER_TIME_UNKNOWN.
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            logger.info("Battery information not available on this system.")
            return {"status": "not_available", "message": "Battery sensors not found on this system."}

        secsleft_str = "Unknown"
        if battery.secsleft == psutil.POWER_TIME_UNLIMITED:
            secsleft_str = "Unlimited (plugged in)"
        elif battery.secsleft == psutil.POWER_TIME_UNKNOWN:
            secsleft_str = "Unknown"
        elif isinstance(battery.secsleft, (int, float)) and battery.secsleft >= 0:
            minsleft = round(battery.secsleft / 60)
            hoursleft = minsleft // 60
            minsleft %= 60
            secsleft_str = f"{hoursleft}h {minsleft}m remaining"
            if not battery.power_plugged and hoursleft == 0 and minsleft < 15 : # Low battery warning
                 secsleft_str += " (Low Battery!)"


        status = {
            "percent": battery.percent,
            "secsleft_description": secsleft_str,
            "power_plugged": battery.power_plugged
        }
        logger.info(f"Battery Status: {status['percent']}% | Time Left: {status['secsleft_description']} | Plugged In: {status['power_plugged']}")
        return status
    except Exception as e:
        logger.error(f"Error getting battery status: {e}", exc_info=True)
        # Check if it's due to battery not being present vs an actual error
        if "battery not found" in str(e).lower() or isinstance(e, NotImplementedError): # some systems might raise NotImplementedError
             logger.info("Battery information not available on this system (exception caught).")
             return {"status": "not_available", "message": "Battery sensors not found or supported on this system."}
        return {"error": f"Could not retrieve battery status: {e}"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    print("--- System Monitor ---")
    print("CPU Usage:", get_cpu_usage())
    import time; time.sleep(0.5) # Give some time for next CPU measure if called rapidly
    print("Memory Usage:", get_memory_usage())
    print("Disk Usage (root/C:):", get_disk_usage("C:\\" if psutil.WINDOWS else "/")) # Adjust path for OS
    # print("Disk Usage (current dir):", get_disk_usage("."))
    print("Battery Status:", get_battery_status())
