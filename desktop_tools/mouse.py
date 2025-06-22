import pyautogui

# PyAutoGUI global settings (FAILSAFE, PAUSE) will use their library defaults.
# If they need to be configurable, DesktopToolDispatcher or WuBuEngine can set them once at startup
# using values from wubu_config.yaml. For now, relying on PyAutoGUI's defaults.
# pyautogui.FAILSAFE = True (default is True)
# pyautogui.PAUSE = 0.1 (default is 0.1)

def mouse_move(x: int, y: int, duration: float = 0.25) -> None:
    """
    Moves the mouse cursor to the specified X, Y coordinates.

    Args:
        x: The target x-coordinate.
        y: The target y-coordinate.
        duration: The time in seconds to spend moving the mouse. Defaults to 0.25.
    """
    try:
        pyautogui.moveTo(x, y, duration=duration)
    except Exception as e:
        print(f"Error moving mouse: {e}")
        raise


def mouse_click(
    x: int = None,
    y: int = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1,
) -> None:
    """
    Performs a mouse click. Can click at current position or specified X, Y coordinates.

    Args:
        x: Optional. The target x-coordinate. If None, clicks at current mouse position.
        y: Optional. The target y-coordinate. If None, clicks at current mouse position.
        button: The mouse button to click ('left', 'middle', 'right'). Defaults to 'left'.
        clicks: The number of times to click. Defaults to 1.
        interval: The time in seconds between clicks. Defaults to 0.1.
    """
    try:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
    except Exception as e:
        print(f"Error clicking mouse: {e}")
        raise


def mouse_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left",
) -> None:
    """
    Drags the mouse from a starting position to an ending position.

    Args:
        start_x: The x-coordinate of the drag start.
        start_y: The y-coordinate of the drag start.
        end_x: The x-coordinate of the drag end.
        end_y: The y-coordinate of the drag end.
        duration: The time in seconds to spend dragging. Defaults to 0.5.
        button: The mouse button to hold down during the drag ('left', 'middle', 'right'). Defaults to 'left'.
    """
    try:
        pyautogui.moveTo(start_x, start_y, duration=0.05) # Quick move to start
        pyautogui.dragTo(end_x, end_y, duration=duration, button=button)
    except Exception as e:
        print(f"Error dragging mouse: {e}")
        raise


def mouse_scroll(amount: int, x: int = None, y: int = None) -> None:
    """
    Scrolls the mouse wheel. Positive amount scrolls up, negative scrolls down.

    Args:
        amount: The number of units to scroll. Positive for up, negative for down.
        x: Optional. The x-coordinate where the scroll should occur. Defaults to current mouse position.
        y: Optional. The y-coordinate where the scroll should occur. Defaults to current mouse position.
    """
    try:
        pyautogui.scroll(amount, x=x, y=y)
    except Exception as e:
        print(f"Error scrolling mouse: {e}")
        raise


if __name__ == "__main__":
    pyautogui.FAILSAFE = True

    try:
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position: {current_x}, {current_y}")
        print("To test mouse functions, uncomment them below and ensure a safe context.")

        # print("Testing move (to current_x + 50, current_y + 50) in 3s...")
        # pyautogui.sleep(3)
        # mouse_move(current_x + 50, current_y + 50, duration=0.5)
        # pyautogui.sleep(1)
        # mouse_move(current_x, current_y, duration=0.5)
        # print("Move test complete.")

        # print("Testing click (left button at current position) in 3s...")
        # pyautogui.sleep(3)
        # mouse_click()
        # print("Click test complete.")

    except pyautogui.FailSafeException:
        print("PyAutoGUI FAILSAFE triggered. Script terminated.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
