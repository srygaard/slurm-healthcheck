"""System firmware and board from DMI; these files are world-readable, unlike the serial numbers beside them."""

from pathlib import Path

from healthcheck.util.file import JsonDict, read_sys

FIELDS = ("bios_vendor", "bios_version", "bios_date", "sys_vendor", "product_name", "board_name")


def collect(root: Path = Path("/sys/class/dmi/id")) -> JsonDict:
    """Each of FIELDS, None where the platform has no DMI (some VMs, ARM boards)."""
    return {field: read_sys(root / field) for field in FIELDS}
