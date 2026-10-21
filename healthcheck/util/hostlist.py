"""Slurm hostlist expressions, e.g. "rc[12501-12531],rg[12501-12503]"."""

import re
from collections.abc import Iterable
from typing import Self


class Hostlist(tuple[str, ...]):
    """
    Unique host names in natural order; str() is the compressed Slurm form. JSON writes it as a list.

    >>> hosts = Hostlist.expand("rg[31501-31503,31601],login1")
    >>> list(hosts)
    ['login1', 'rg31501', 'rg31502', 'rg31503', 'rg31601']
    >>> print(hosts)
    login1,rg[31501-31503,31601]
    """

    @staticmethod
    def _natural_key(name: str) -> tuple[str, int, str]:
        """Sort rg9 before rg10."""
        if match := re.fullmatch(r"(.*?)(\d+)", name):
            return match.group(1), int(match.group(2)), name
        return name, -1, name

    @staticmethod
    def _split_top_level(expr: str) -> list[str]:
        """Split on commas outside brackets."""
        parts = []
        depth = 0
        start = 0
        for index, char in enumerate(expr):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(expr[start:index])
                start = index + 1
        parts.append(expr[start:])
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _expand_range(spec: str) -> list[str]:
        out = []
        for item in spec.split(","):
            lo, _, hi = item.partition("-")
            if not hi:
                out.append(lo)
                continue
            if int(hi) < int(lo):
                raise ValueError(f"Descending range {item!r}")
            width = len(lo)  # keep zero padding, e.g. 001-010
            out.extend(f"{number:0{width}d}" for number in range(int(lo), int(hi) + 1))
        return out

    def __new__(cls, names: Iterable[str] = ()) -> Self:
        return super().__new__(cls, sorted(set(names), key=cls._natural_key))

    @classmethod
    def expand(cls, expr: str) -> Self:
        names: list[str] = []
        for part in cls._split_top_level(expr):
            # Expand the first bracket group, then recurse for any that follow (e.g. "r[1-2]n[1-2]").
            if match := re.search(r"\[([^\]]+)\]", part):
                head = part[: match.start()]
                tail = part[match.end() :]
                for value in cls._expand_range(match.group(1)):
                    names.extend(cls.expand(f"{head}{value}{tail}"))
            else:
                names.append(part)
        return cls(names)

    def compress(self) -> str:
        groups: dict[tuple[str, int], list[int]] = {}
        out: list[str] = []
        for name in self:
            if match := re.fullmatch(r"(.*?)(\d+)", name):
                groups.setdefault((match.group(1), len(match.group(2))), []).append(int(match.group(2)))
            else:
                out.append(name)

        for (prefix, width), numbers in groups.items():
            if len(numbers) == 1:
                out.append(f"{prefix}{numbers[0]:0{width}d}")
                continue
            runs = []
            start = 0
            for index in range(1, len(numbers) + 1):
                if index == len(numbers) or numbers[index] != numbers[index - 1] + 1:
                    runs.append((numbers[start], numbers[index - 1]))
                    start = index
            ranges = [f"{lo:0{width}d}" if lo == hi else f"{lo:0{width}d}-{hi:0{width}d}" for lo, hi in runs]
            out.append(f"{prefix}[{','.join(ranges)}]")
        return ",".join(sorted(out))

    def __str__(self) -> str:
        return self.compress()

    def __repr__(self) -> str:
        return f"Hostlist.expand({self.compress()!r})"
