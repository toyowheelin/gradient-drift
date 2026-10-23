"""One program for both ways of starting it on Windows. It is built as a console program, so
from a terminal it behaves like one. Double-clicked, Windows opens a console just for it; setup
then hides that console and opens the window instead."""
import sys


def _kernel32():
    import ctypes
    return ctypes.windll.kernel32


def double_clicked():
    """True when the console was made just for this program (Explorer started it), False when it
    shares one with a terminal, and always False off Windows."""
    if sys.platform != 'win32':
        return False
    import ctypes
    procs = (ctypes.c_uint * 4)()
    return _kernel32().GetConsoleProcessList(procs, 4) == 1


def hide():
    import ctypes
    window = _kernel32().GetConsoleWindow()
    if window:
        ctypes.windll.user32.ShowWindow(window, 0)   # SW_HIDE


def show():
    import ctypes
    window = _kernel32().GetConsoleWindow()
    if window:
        ctypes.windll.user32.ShowWindow(window, 5)   # SW_SHOW


def detach():
    """Let go of the console so it closes (also covers Windows Terminal, where hiding does not)."""
    _kernel32().FreeConsole()
    sys.stdout = sys.stderr = None   # nothing to print to any more


def error_box(text):
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, text, 'Gradient Drift Setup - error', 0x10)   # MB_ICONERROR
