"""Pipeline runners (shell + Python helper scripts).

Package marker: without it this directory is a PEP 420 namespace package and
mypy resolves e.g. ``_load_comparison`` under two different module names
(``_load_comparison`` vs ``runners._load_comparison``), which makes
``mypy runners/`` abort with "Source file found twice under different module
names". With the marker, every module has one stable name (``runners.*``) so
mypy, tests, and the cross-import in run_harmbench.py all agree.
"""
