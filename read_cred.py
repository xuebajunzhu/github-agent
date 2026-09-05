"""Read the GitHub credential stored by Git Credential Manager via CredRead."""
import ctypes
import sys

advapi32 = ctypes.WinDLL("advapi32")
kernel32 = ctypes.WinDLL("kernel32")

CRED_TYPE_GENERIC = 1


class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
        ("TargetName", ctypes.c_wchar_p),
        ("Comment", ctypes.c_wchar_p),
        ("LastWritten", ctypes.c_ulonglong),
        ("CredentialBlobSize", ctypes.c_ulong),
        ("CredentialBlob", ctypes.c_void_p),
        ("Persist", ctypes.c_ulong),
        ("AttributeCount", ctypes.c_ulong),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", ctypes.c_wchar_p),
        ("UserName", ctypes.c_wchar_p),
    ]


targets = sys.argv[1:] or ["git:https://github.com", "gh:github.com"]
for target in targets:
    cred = CREDENTIAL()
    if not advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(cred)):
        print(f"{target}: not readable (error {kernel32.GetLastError()})")
        continue
    blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize) if cred.CredentialBlobSize else b""
    try:
        secret = blob.decode("utf-8")
    except UnicodeDecodeError:
        secret = blob.decode("utf-16-le", errors="ignore").rstrip("\x00")
    print(f"{target}: user={cred.UserName} secret_len={len(secret)} head={secret[:4]}...")
