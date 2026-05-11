Dim shell
Set shell = CreateObject("WScript.Shell")
shell.Run "cmd.exe /c ""C:\projects\genshape-worker-3090\.venv\Scripts\pythonw.exe"" ""C:\projects\genshape-worker-3090\tray.py""", 0, False
Set shell = Nothing
