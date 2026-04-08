"""Run all test suites."""
import subprocess
import sys
import os

tests = [
    "tests/test_environment.py",
    "tests/test_server.py",
    "tests/test_inference.py",
    "tests/test_stdout_compliance.py",
    "tests/test_tools.py",
    "tests/test_temporal_unit.py",
    "tests/test_temporal.py",
    "tests/test_rewards_graders.py",
    "tests/test_enhancements.py",
    "tests/test_scenarios_graders.py",
]

failed = []
for t in tests:
    print("\n" + "=" * 60)
    print(f"Running {t}")
    print("=" * 60)
    env = {**os.environ, "PYTHONPATH": ".", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, t], env=env)
    if r.returncode != 0:
        failed.append(t)

print("\n" + "=" * 60)
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
