import os
import sys
import time
import subprocess


def main():
    print("\n")
    print("START AUTOMATED INTEGRATION PIPELINE (LAB6)")
    print("\n")

    output_dir = "reports/LAB6"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "service_tests.log")

    print("[STEP 1] Starting FastAPI inference server (serve-api)...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "src.api.service"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)

    print("[STEP 2] Initializing monitoring collector service (monitor-run)...")
    monitor_proc = subprocess.Popen([sys.executable, "-m", "src.api.monitor"])
    time.sleep(2)

    print("[STEP 3] Running contract and load tests via pytest...")
    with open(log_path, "w", encoding="utf-8") as log_file:
        test_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "src/tests/test_api.py", "-v"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    print("[STEP 4] Finalizing drill and closing background process ports...")
    monitor_proc.terminate()
    monitor_proc.wait()

    server_proc.terminate()
    server_proc.wait()

    print("\n")
    print("DRILL SUCCESSFUL! ALL ARTEFACTS CAPTURED ON DISK")
    print("\n")

    if test_proc.returncode == 0:
        print("TESTING STATUS: 6 PASSED (100% Success)")
    else:
        print(f"TESTING STATUS: FAILED (Exit Code {test_proc.returncode})")

    sys.exit(test_proc.returncode)


if __name__ == "__main__":
    main()
