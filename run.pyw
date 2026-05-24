"""Silent launcher — invoked with pythonw.exe so no console appears."""

from support_role.main import run

if __name__ == "__main__":
    raise SystemExit(run(debug=False))
