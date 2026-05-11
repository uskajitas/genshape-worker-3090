' VBS entry that fully detaches the worker from any console.
' Same pattern as the i7's start-server.vbs — VBS launches PS1, PS1 does the work.
' WindowStyle=0 (hidden), WaitOnReturn=False (fire and forget).
Set sh = CreateObject("WScript.Shell")
script = "powershell -NoProfile -ExecutionPolicy Bypass -File """ & _
         CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & _
         "\start-worker.ps1"""
sh.Run script, 0, False
