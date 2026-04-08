"""Run all test suites."""
import subprocess
import sys
import os

tests = [
    "tests/test_environment.py",
    "tests/test_server.py",
    "tests/test_inference.py",
]

failed = []
for t in tests:
    print("\n" + "=" * 60)
    print(f"Running {t}")
    print("=" * 60)
    env = {**os.environ, "PYTHONPATH": "."}
    r = subprocess.run([sys.executable, t], env=env)
    if r.returncode != 0:
        failed.append(t)

print("\n" + "=" * 60)
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
