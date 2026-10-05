"""Driver scripts that run *inside the MITIM interpreter*.

They are copied into the run directory and executed there, so they import
MITIM and the standard library only -- never VAFT. Each reads the JSON file
named by its first argument and writes ``result.json`` beside itself, with
``status`` ``"ok"`` or ``"error"``.
"""
