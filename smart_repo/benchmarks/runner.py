"""Benchmark runner — executes all 12 tests and reports results."""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    name: str
    passed: bool
    duration_ms: float
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Discovers and runs all benchmark tests.

    Tests are async functions named `test_*` in the test_cases module.
    """

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self.workspace_dir = workspace_dir or Path.cwd()
        self.results: list[BenchmarkResult] = []

    def discover(self) -> list[Callable[..., Awaitable[bool]]]:
        """Discover all benchmark test functions."""
        from smart_repo.benchmarks import test_cases
        import inspect

        tests = []
        for name in dir(test_cases):
            if name.startswith("test_benchmark_"):
                fn = getattr(test_cases, name)
                if callable(fn) and inspect.iscoroutinefunction(fn):
                    tests.append(fn)
        return sorted(tests, key=lambda f: f.__name__)

    async def run_all(self) -> list[BenchmarkResult]:
        """Run all discovered benchmarks."""
        tests = self.discover()
        self.results = []

        console = Console()
        console.print(f"\n[bold]Running {len(tests)} benchmarks...[/bold]\n")

        for test_fn in tests:
            name = test_fn.__name__.replace("test_benchmark_", "").replace("_", " ").title()
            console.print(f"  Running: {name}...", end=" ")

            start = time.time()
            try:
                result = await test_fn(self.workspace_dir)
                duration = (time.time() - start) * 1000
                if isinstance(result, bool):
                    passed = result
                elif isinstance(result, dict):
                    passed = result.get("passed", False)
                else:
                    passed = bool(result)

                br = BenchmarkResult(
                    name=name,
                    passed=passed,
                    duration_ms=duration,
                    details=result if isinstance(result, dict) else {},
                )
                console.print("[green][PASS][/green]" if passed else "[red][FAIL][/red]")
            except Exception as e:
                duration = (time.time() - start) * 1000
                br = BenchmarkResult(
                    name=name,
                    passed=False,
                    duration_ms=duration,
                    error=f"{type(e).__name__}: {e}",
                )
                console.print(f"[red][ERROR]: {e}[/red]")

            self.results.append(br)

        return self.results

    def print_summary(self) -> None:
        """Print a summary table of all benchmark results."""
        console = Console()

        table = Table(title="SmartRepo Benchmark Results — 13 Code Benchmark Tests")
        table.add_column("#", style="dim", width=4)
        table.add_column("Test", style="cyan")
        table.add_column("Result", justify="center")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Notes")

        passed = 0
        for i, result in enumerate(self.results, 1):
            status = "[green][PASS][/green]" if result.passed else "[red][FAIL][/red]"
            notes = result.error[:60] if result.error else ""
            table.add_row(
                str(i),
                result.name,
                status,
                f"{result.duration_ms:.1f}",
                notes,
            )
            if result.passed:
                passed += 1

        console.print()
        console.print(table)
        console.print()

        total = len(self.results)
        if passed == total:
            console.print(
                f"[bold green][PASS] All {total} benchmarks passed! (100% pass rate)[/bold green]"
            )
        else:
            failed = total - passed
            pct = (passed / total) * 100 if total > 0 else 0
            console.print(
                f"[bold yellow]{passed}/{total} passed ({pct:.0f}%) — {failed} failed[/bold yellow]"
            )

        return passed, total


async def main():
    """Run benchmarks from the CLI."""
    import argparse
    parser = argparse.ArgumentParser(description="SmartRepo Benchmark Runner")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace directory for tests")
    parser.add_argument("--filter", "-f", default="", help="Filter tests by name")
    args = parser.parse_args()

    runner = BenchmarkRunner(workspace_dir=Path(args.workspace))
    test_funcs = runner.discover()

    if args.filter:
        test_funcs = [f for f in test_funcs if args.filter.lower() in f.__name__.lower()]

    # Override discover to only run filtered tests
    original_discover = runner.discover
    runner.discover = lambda: test_funcs

    results = await runner.run_all()
    passed, total = runner.print_summary()

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)
