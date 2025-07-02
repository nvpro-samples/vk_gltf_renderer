import argparse
import subprocess
import logging
from pathlib import Path
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Predefined executables and their arguments
EXECUTABLES_WITH_ARGS = [
    ("vk_gltf_renderer", ["--headless", "shader_ball.gltf", "env3.hdr", "--envSystem", "1", "--frames", "10"]),

]

class ReturnCode(Enum):
    SUCCESS = 0
    TEST_ERROR = 1
    ENVIRONMENT_ERROR = 2

def run_command(commands):
    """Run a command and handle potential errors."""
    logger.info(f"Running command: {' '.join(commands)}")
    try:
        result = subprocess.run(
            commands,
            shell=isinstance(commands, str),
            check=True,
            text=True,
            capture_output=True,
        )
        logger.info(f"Command output:\n{result.stdout}\n{result.stderr}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred: {e}")
        logger.error(f"Command output:\n{e.stdout}\n{e.stderr}")
        return e.returncode

def extract_testing_time(log_file):
    """Extract testing time from the last line of the log file.
    Example log line: '[00:000:01.993]  -> 84.683 ms'
    Returns the value after '->' or 'N/A' if not found."""
    if log_file.exists():
        content = log_file.read_text()
        lines = content.splitlines()
        if lines:  # if there are any lines
            last_line = lines[-1].strip()
            if "->" in last_line:
                try:
                    time_part = last_line.split("->")[1].strip()
                    return time_part
                except IndexError:
                    pass
    return "N/A"

def test_executable(test_dir, executable, args):
    """Test a single executable and return its results."""
    try:
        executable_path = test_dir / executable
        # if not executable_path.exists():
        #     logger.error(f"Executable not found: {executable_path}")
        #     return (executable, ReturnCode.ENVIRONMENT_ERROR.value, "N/A")
            
        return_code = run_command([str(executable_path)] + args)
        log_file = test_dir / Path(f"log_{executable}.txt")
        testing_time = extract_testing_time(log_file)
        
        return (
            executable,
            ReturnCode.SUCCESS.value if return_code == 0 else ReturnCode.TEST_ERROR.value,
            testing_time
        )
    except Exception as e:
        logger.error(f"Error testing {executable}: {e}")
        return (executable, ReturnCode.TEST_ERROR.value, "N/A")

def test_all_executables():
    """Test all predefined executables."""
    # Install directory for executables
    test_dir = Path("_install")
    if not test_dir.exists():
        logger.error(f"Test directory '{test_dir}' does not exist.")
        return [], ReturnCode.ENVIRONMENT_ERROR
    
    results = []
    overall_status = ReturnCode.SUCCESS
    
    for executable, args in EXECUTABLES_WITH_ARGS:
        logger.info(f"\nTesting: {executable}")
        result = test_executable(test_dir, executable, args)
        results.append(result)
        if result[1] != ReturnCode.SUCCESS.value:
            overall_status = ReturnCode.TEST_ERROR
    
    return results, overall_status

def print_report(results):
    """Print a formatted report of test results."""
    logger.info("\nTest Results:")
    logger.info("-" * 80)
    logger.info("{:<35} | {:<12} | {:<8}".format("Executable", "Status", "Time"))
    logger.info("-" * 80)
    
    for executable, return_code, testing_time in results:
        status = "SUCCESS" if return_code == ReturnCode.SUCCESS.value else "FAILED"
        logger.info("{:<35} | {:<12} | {:<8}".format(
            executable, status, testing_time
        ))
    logger.info("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Test executables and generate reports.")
    parser.add_argument("--test", action="store_true", help="Run tests on all predefined executables")
    
    args = parser.parse_args()
    
    if args.test:
        results, status = test_all_executables()
        print_report(results)
        return status
    else:
        parser.error("--test flag is required")

if __name__ == "__main__":
    try:
        result = main()
        exit(result.value)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(ReturnCode.TEST_ERROR.value)