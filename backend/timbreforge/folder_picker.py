import ctypes as c
from pathlib import Path
import sys
from threading import Lock
from uuid import UUID

from .jobs import Conflict
from .worker import JobError


_dialog = Lock()


def pick_folder(initial):
    if sys.platform != "win32":
        raise JobError("原生目录选择窗口仅支持 Windows，请手动输入目录路径")
    if not _dialog.acquire(blocking=False):
        raise Conflict("请先完成已打开的目录选择")
    try:
        return _pick_windows(initial)
    except OSError:
        raise JobError("无法打开 Windows 目录选择窗口，请手动输入目录路径") from None
    finally:
        _dialog.release()


def _pick_windows(initial):
    ole, shell = c.windll.ole32, c.windll.shell32
    ptr = c.c_void_p
    dword = c.c_uint32

    def guid(value):
        return (c.c_ubyte * 16).from_buffer_copy(UUID(value).bytes_le)

    def checked(result):
        if result < 0:
            raise OSError(f"HRESULT {result & 0xFFFFFFFF:08X}")

    def call(obj, slot, restype, argtypes=(), *args):
        vtable = c.cast(obj, c.POINTER(c.POINTER(ptr))).contents
        return c.WINFUNCTYPE(restype, ptr, *argtypes)(vtable[slot])(obj, *args)

    def release(obj):
        if obj:
            call(obj, 2, dword)  # IUnknown::Release

    checked(ole.CoInitializeEx(None, 2))  # COINIT_APARTMENTTHREADED
    dialog, start, item, text = ptr(), ptr(), ptr(), ptr()
    try:
        clsid = guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")
        interface = guid("D57C7288-D4AD-4768-BE02-9D969532D960")
        checked(ole.CoCreateInstance(c.byref(clsid), None, 1, c.byref(interface), c.byref(dialog)))
        # Slots in IFileDialog, following IUnknown and IModalWindow.
        options = dword()
        checked(call(dialog, 10, c.c_long, (c.POINTER(dword),), c.byref(options)))  # GetOptions
        # FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM
        checked(call(dialog, 9, c.c_long, (dword,), options.value | 0x60))  # SetOptions
        checked(call(dialog, 17, c.c_long, (c.c_wchar_p,), "选择训练录音目录"))  # SetTitle
        directory = Path(initial).expanduser() if initial else None
        if directory is not None and directory.is_dir():
            shell_item = guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")
            found = shell.SHCreateItemFromParsingName(c.c_wchar_p(str(directory.resolve())), None,
                                                      c.byref(shell_item), c.byref(start))
            if found >= 0:
                checked(call(dialog, 12, c.c_long, (ptr,), start))  # SetFolder
        foreground = c.windll.user32.GetForegroundWindow
        foreground.restype = ptr
        result = call(dialog, 3, c.c_long, (ptr,), foreground())  # Show
        if result & 0xFFFFFFFF == 0x800704C7:  # HRESULT_FROM_WIN32(ERROR_CANCELLED)
            return None
        checked(result)
        checked(call(dialog, 20, c.c_long, (c.POINTER(ptr),), c.byref(item)))  # GetResult
        # IShellItem::GetDisplayName(SIGDN_FILESYSPATH)
        checked(call(item, 5, c.c_long, (dword, c.POINTER(ptr)), 0x80058000, c.byref(text)))
        return c.wstring_at(text)
    finally:
        if text:
            ole.CoTaskMemFree(text)
        for obj in (item, start, dialog):
            release(obj)
        ole.CoUninitialize()
