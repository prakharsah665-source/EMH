"""
PASS / FAIL check records shared by every validation stage.

A CheckResult is one row of the final report: what was checked, what
the API (or the derivation from it) said the value must be, what was
actually observed, and the verdict. Tests collect their rows through
the session registry and assert on them, so the pytest outcome and
the report can never disagree.
"""

from dataclasses import asdict, dataclass


PASS = "PASS"
FAIL = "FAIL"
NA = "N/A"


@dataclass
class CheckResult:
    category: str
    name: str
    status: str
    expected: str = ""
    actual: str = ""
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status != FAIL

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        parts = [f"[{self.status}] {self.category}: {self.name}"]
        if self.expected or self.actual:
            parts.append(
                f"expected={self.expected!s} actual={self.actual!s}"
            )
        if self.detail:
            parts.append(self.detail)
        return " | ".join(parts)


def check(
    category: str,
    name: str,
    ok: bool,
    expected: object = "",
    actual: object = "",
    detail: str = "",
) -> CheckResult:
    return CheckResult(
        category=category,
        name=name,
        status=PASS if ok else FAIL,
        expected=str(expected),
        actual=str(actual),
        detail=detail,
    )


def not_applicable(category: str, name: str, detail: str) -> CheckResult:
    return CheckResult(
        category=category, name=name, status=NA, detail=detail
    )


class CheckRegistry:
    """Session-wide collection of every check, in execution order."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, results: list[CheckResult]) -> None:
        self.results.extend(results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    def counts(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NA: 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        counts["total"] = len(self.results)
        return counts

    @property
    def overall(self) -> str:
        return FAIL if self.failures else PASS


def assert_all_pass(results: list[CheckResult], heading: str) -> None:
    """
    Print every check (captured into the report's per-test output)
    and fail the test with only the FAILED rows if any.
    """

    print(f"\n{heading}")
    print("-" * len(heading))
    for result in results:
        print(result.line())

    failed = [r for r in results if r.status == FAIL]
    if failed:
        lines = [f"{len(failed)} of {len(results)} checks FAILED:"]
        for result in failed:
            lines.append(f"  - {result.line()}")
        raise AssertionError("\n".join(lines))
