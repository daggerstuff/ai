import pathlib

ROUTES = pathlib.Path(__file__).resolve().parents[2] / "api" / "mcp_server" / "routes.py"

GENERIC_CONFLICT = "The requested retain operation could not be completed due to a scope conflict."
GENERIC_ACCESS = "The requested document could not be accessed."


def _retain_region(src: str) -> str:
    start = src.find("def _prepare_foresight_retain_items")
    if start == -1:
        return src
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt] if nxt != -1 else src[start:]


def test_security_fix_present_in_routes():
    src = ROUTES.read_text()
    assert GENERIC_CONFLICT in src
    assert GENERIC_ACCESS in src


def test_retain_handler_does_not_leak_raw_exception():
    src = ROUTES.read_text()
    region = _retain_region(src)
    assert "str(exc)" not in region
    assert "logger.error" in region
