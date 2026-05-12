import os
import time
import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_file_path(file: str) -> str:
    """Returns the absolute path to the file in the temp directory."""
    return str(Path.cwd() / "temp" / file)


@contextlib.contextmanager
def acquire_file_lock(file_path: str, timeout: int = 10):
    """
    Context manager for atomic file locking.
    Waits up to `timeout` seconds to acquire the lock.
    """
    lock_path = file_path + ".lock"
    start_time = time.time()

    while True:
        try:
            # os.O_CREAT | os.O_EXCL ensures atomic creation.
            # It will instantly raise FileExistsError if the lock is already there.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, b"locked")
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Could not acquire lock for {file_path} within {timeout}s."
                )
            time.sleep(0.2)

    try:
        # Yield control back to the code inside the 'with' block
        yield
    finally:
        # The lock is guaranteed to be released when the 'with' block finishes,
        # even if an error crashes the code inside the block.
        try:
            os.remove(lock_path)
        except OSError:
            pass


def safe_write_to_file(filename: str, content: str) -> str:
    """
    A combined function that handles paths, directories, locking, and writing safely.
    Returns a success or error message that can be passed back to the agent.
    """
    file_to_use = make_file_path(filename)

    # 1. Ensure directory exists
    os.makedirs(os.path.dirname(file_to_use), exist_ok=True)

    # 2. Acquire lock and write safely
    try:
        with acquire_file_lock(file_to_use):
            with open(file_to_use, "w") as f:
                f.write(content)
        logger.info(f"Successfully wrote to {filename}")
        return f"Successfully saved to {filename}"
    except TimeoutError as e:
        logger.warning(str(e))
        return f"Error: The file {filename} is currently locked by another process. Try again later."
    except Exception as e:
        logger.error(f"Failed to write to {filename}: {e}")
        return f"Error: Failed to save file {filename}: {str(e)}"
