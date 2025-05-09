🔹 Game Control Logic

Not started: Waits for "hands joined" gesture.

Started: Reads posture and position to simulate:

Left/Right: pyautogui.press('left'/'right')

Jump: pyautogui.press('up')

Crouch: pyautogui.press('down')

Start game or jump again: pyautogui.click() or pyautogui.press('space')