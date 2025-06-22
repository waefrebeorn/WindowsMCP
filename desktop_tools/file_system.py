import logging
from pathlib import Path
from typing import List, Dict, Union, Optional

logger = logging.getLogger(__name__)

# Define a maximum file size for reading to prevent memory issues with large files
MAX_READ_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def list_directory(path_str: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Lists the contents (files and subdirectories) of a given directory path.

    Args:
        path_str: The absolute or relative path to the directory.

    Returns:
        A dictionary containing:
        - "path": The absolute path of the listed directory.
        - "contents": A list of dictionaries, each representing an item in the directory
                      with "name" and "type" ('file' or 'directory').
        - "error": An error message if the path is invalid or inaccessible.
    """
    try:
        path = Path(
            path_str
        ).resolve()  # Resolve to absolute path and handle symlinks potentially

        if not path.exists():
            return {"error": f"Path does not exist: {path_str}"}
        if not path.is_dir():
            return {"error": f"Path is not a directory: {path_str}"}

        items = []
        for item in path.iterdir():
            item_type = "directory" if item.is_dir() else "file"
            items.append({"name": item.name, "type": item_type})

        logger.info(f"Listed directory '{path}'. Found {len(items)} items.")
        return {"path": str(path), "contents": items}

    except PermissionError:
        logger.error(f"Permission denied for path: {path_str}")
        return {"error": f"Permission denied for path: {path_str}"}
    except Exception as e:
        logger.error(f"Error listing directory '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while listing directory: {e}"}


def read_text_file(path_str: str, max_chars: Optional[int] = None) -> Dict[str, str]:
    """
    Reads the content of a text file.

    Args:
        path_str: The absolute or relative path to the text file.
        max_chars: Optional. Maximum number of characters to read from the beginning of the file.
                   If None, attempts to read up to MAX_READ_FILE_SIZE_BYTES.

    Returns:
        A dictionary containing:
        - "path": The absolute path of the read file.
        - "content": The content of the file (potentially truncated).
        - "error": An error message if the path is invalid, not a file, too large, or unreadable.
        - "warning": A warning message if the content was truncated.
    """
    try:
        path = Path(path_str).resolve()

        if not path.exists():
            return {"error": f"File does not exist: {path_str}"}
        if not path.is_file():
            return {"error": f"Path is not a file: {path_str}"}

        file_size = path.stat().st_size
        if file_size > MAX_READ_FILE_SIZE_BYTES and max_chars is None:
            warning_msg = (
                f"File is large ({file_size} bytes). "
                f"Reading only the first {MAX_READ_FILE_SIZE_BYTES // 1024}KB."
            )
            logger.warning(warning_msg)
            # Read only up to the defined max size for very large files if no specific max_chars
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_READ_FILE_SIZE_BYTES)
            return {"path": str(path), "content": content, "warning": warning_msg}

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if max_chars is not None and max_chars > 0:
                content = f.read(max_chars)
                warning = (
                    "Content truncated to max_chars."
                    if len(content) == max_chars and file_size > max_chars
                    else None
                )
                logger.info(
                    f"Read {len(content)} characters from file '{path}'. Max_chars was {max_chars}."
                )
                return {"path": str(path), "content": content, "warning": warning}
            else:  # No max_chars or invalid max_chars, read full file (up to internal limit already checked)
                content = f.read()
                logger.info(f"Read file '{path}'. Length: {len(content)} chars.")
                return {"path": str(path), "content": content}

    except PermissionError:
        logger.error(f"Permission denied for file: {path_str}")
        return {"error": f"Permission denied for file: {path_str}"}
    except UnicodeDecodeError:
        logger.error(
            f"Cannot decode file as UTF-8 text: {path_str}. It might be a binary file."
        )
        return {
            "error": f"File is likely not a text file or has an unsupported encoding: {path_str}"
        }
    except Exception as e:
        logger.error(f"Error reading file '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while reading file: {e}"}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    logger.info("File System Module Example")

    # Create some dummy files and directories for testing
    test_dir = Path("./test_fs_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.txt").write_text("This is file1.\nHello world.")
    (test_dir / "file2.log").write_text("Log entry 1\nLog entry 2")
    (test_dir / "subdir").mkdir(exist_ok=True)
    (test_dir / "subdir" / "nested_file.txt").write_text("Nested content.")

    # Test list_directory
    print("\n--- Listing current directory ---")
    # current_dir_contents = list_directory(".") # Use a known directory for more consistent testing
    test_dir_contents = list_directory(str(test_dir))
    if "error" in test_dir_contents:
        print(f"Error: {test_dir_contents['error']}")
    else:
        print(f"Contents of {test_dir_contents['path']}:")
        for item in test_dir_contents.get("contents", []):
            print(f"  - {item['name']} ({item['type']})")

    print("\n--- Listing non-existent directory ---")
    non_existent_contents = list_directory("./non_existent_dir_test_fs")
    if "error" in non_existent_contents:
        print(f"Correctly handled error: {non_existent_contents['error']}")
    else:
        print("Error: Expected an error for non-existent directory.")

    # Test read_text_file
    file1_path = str(test_dir / "file1.txt")
    print(f"\n--- Reading file: {file1_path} ---")
    file1_data = read_text_file(file1_path)
    if "error" in file1_data:
        print(f"Error: {file1_data['error']}")
    else:
        print(f"Content of {file1_data['path']}:\n---\n{file1_data['content']}\n---")
        if "warning" in file1_data and file1_data["warning"]:
            print(f"Warning: {file1_data['warning']}")

    print(f"\n--- Reading file with max_chars: {file1_path} (10 chars) ---")
    file1_partial_data = read_text_file(file1_path, max_chars=10)
    if "error" in file1_partial_data:
        print(f"Error: {file1_partial_data['error']}")
    else:
        print(f"Content (partial): {file1_partial_data['content']}")
        if "warning" in file1_partial_data and file1_partial_data["warning"]:
            print(f"Warning: {file1_partial_data['warning']}")

    # Test reading a non-text file (e.g., a directory treated as a file)
    print(f"\n--- Attempting to read directory as text file: {test_dir} ---")
    dir_as_file_data = read_text_file(str(test_dir))
    if "error" in dir_as_file_data:
        print(f"Correctly handled error: {dir_as_file_data['error']}")
    else:
        print(
            "Error: Expected an error when trying to read a directory as a text file."
        )

    # Clean up dummy files and directory
    try:
        (test_dir / "subdir" / "nested_file.txt").unlink()
        (test_dir / "subdir").rmdir()
        (test_dir / "file1.txt").unlink()
        (test_dir / "file2.log").unlink()
        test_dir.rmdir()
        logger.info("Cleaned up test directory and files.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("File system example finished.")


# --- New File System Operations ---

def write_text_file(path_str: str, content: str, overwrite: bool = False) -> Dict[str, str]:
    """
    Writes text content to a file. By default, it will not overwrite an existing file
    unless 'overwrite' is set to True.

    Args:
        path_str: The path to the file.
        content: The text content to write.
        overwrite: If True, overwrite the file if it exists. Defaults to False.

    Returns:
        A dictionary with "path" and "message" on success, or "error".
    """
    try:
        path = Path(path_str).resolve()
        path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

        if path.exists() and not overwrite:
            return {"error": f"File '{path}' already exists and overwrite is False."}
        if path.is_dir():
            return {"error": f"Path '{path}' is a directory, cannot write a file here."}

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote to file '{path}'. Overwrite: {overwrite}.")
        return {"path": str(path), "message": f"Content written to '{path}'."}
    except PermissionError:
        logger.error(f"Permission denied for writing to file: {path_str}")
        return {"error": f"Permission denied for writing to file: {path_str}"}
    except Exception as e:
        logger.error(f"Error writing to file '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while writing to file: {e}"}

def append_text_to_file(path_str: str, content: str) -> Dict[str, str]:
    """
    Appends text content to an existing file. Creates the file if it doesn't exist.

    Args:
        path_str: The path to the file.
        content: The text content to append.

    Returns:
        A dictionary with "path" and "message" on success, or "error".
    """
    try:
        path = Path(path_str).resolve()
        path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

        if path.is_dir():
            return {"error": f"Path '{path}' is a directory, cannot append to it."}

        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully appended to file '{path}'.")
        return {"path": str(path), "message": f"Content appended to '{path}'."}
    except PermissionError:
        logger.error(f"Permission denied for appending to file: {path_str}")
        return {"error": f"Permission denied for appending to file: {path_str}"}
    except Exception as e:
        logger.error(f"Error appending to file '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while appending to file: {e}"}

def create_directory(path_str: str) -> Dict[str, str]:
    """
    Creates a new directory.

    Args:
        path_str: The path for the new directory.

    Returns:
        A dictionary with "path" and "message" on success, or "error".
    """
    try:
        path = Path(path_str).resolve()
        if path.exists():
            if path.is_dir():
                return {"path": str(path), "message": f"Directory '{path}' already exists."}
            else:
                return {"error": f"Path '{path}' already exists and is a file."}

        path.mkdir(parents=True, exist_ok=True) # exist_ok=True means it won't error if dir exists
                                                # but we checked above so it's more for parents
        logger.info(f"Successfully created directory '{path}'.")
        return {"path": str(path), "message": f"Directory '{path}' created."}
    except PermissionError:
        logger.error(f"Permission denied for creating directory: {path_str}")
        return {"error": f"Permission denied for creating directory: {path_str}"}
    except Exception as e:
        logger.error(f"Error creating directory '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while creating directory: {e}"}

def delete_file_or_directory(path_str: str) -> Dict[str, str]:
    """
    Deletes a file or a directory (recursively if it's a directory).

    Args:
        path_str: The path to the file or directory to delete.

    Returns:
        A dictionary with "path" and "message" on success, or "error".
    """
    try:
        path = Path(path_str).resolve()
        if not path.exists():
            return {"error": f"Path '{path}' does not exist. Nothing to delete."}

        if path.is_file():
            path.unlink()
            logger.info(f"Successfully deleted file '{path}'.")
            return {"path": str(path), "message": f"File '{path}' deleted."}
        elif path.is_dir():
            import shutil
            shutil.rmtree(path) # Deletes directory and all its contents
            logger.info(f"Successfully deleted directory '{path}' and its contents.")
            return {"path": str(path), "message": f"Directory '{path}' and its contents deleted."}
        else:
            # This case should be rare (e.g. broken symlink, other special file types)
            return {"error": f"Path '{path}' is neither a file nor a directory. Deletion aborted."}

    except PermissionError:
        logger.error(f"Permission denied for deleting: {path_str}")
        return {"error": f"Permission denied for deleting: {path_str}"}
    except Exception as e:
        logger.error(f"Error deleting '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while deleting: {e}"}

def move_file_or_directory(source_path_str: str, destination_path_str: str) -> Dict[str, str]:
    """
    Moves a file or directory from a source path to a destination path.

    Args:
        source_path_str: The path of the file or directory to move.
        destination_path_str: The path to move the item to. If it's a directory,
                              the source item will be moved into it. If it's a full path
                              (including new name), it will be moved/renamed.

    Returns:
        A dictionary with "source_path", "destination_path", and "message" on success, or "error".
    """
    try:
        source_path = Path(source_path_str).resolve()
        destination_path = Path(destination_path_str).resolve()

        if not source_path.exists():
            return {"error": f"Source path '{source_path}' does not exist."}

        # Ensure destination parent directory exists
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # If destination is an existing directory, move source into it
        if destination_path.is_dir():
            final_destination = destination_path / source_path.name
        else: # Destination is a full path (potentially renaming)
            final_destination = destination_path

        if final_destination.exists():
             return {"error": f"Destination path '{final_destination}' already exists."}


        import shutil
        shutil.move(str(source_path), str(final_destination))
        logger.info(f"Successfully moved '{source_path}' to '{final_destination}'.")
        return {
            "source_path": str(source_path),
            "destination_path": str(final_destination),
            "message": f"Moved '{source_path}' to '{final_destination}'.",
        }
    except PermissionError:
        logger.error(f"Permission denied for moving '{source_path_str}' to '{destination_path_str}'")
        return {"error": f"Permission denied for moving operation."}
    except Exception as e:
        logger.error(f"Error moving '{source_path_str}' to '{destination_path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while moving: {e}"}

def copy_file_or_directory(source_path_str: str, destination_path_str: str) -> Dict[str, str]:
    """
    Copies a file or directory from a source path to a destination path.

    Args:
        source_path_str: The path of the file or directory to copy.
        destination_path_str: The path to copy the item to. If it's an existing directory,
                              the source item will be copied into it. If it's a full path
                              (including new name), it will be copied/renamed.

    Returns:
        A dictionary with "source_path", "destination_path", and "message" on success, or "error".
    """
    try:
        source_path = Path(source_path_str).resolve()
        destination_path = Path(destination_path_str).resolve()

        if not source_path.exists():
            return {"error": f"Source path '{source_path}' does not exist."}

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        if source_path.is_file():
            # If destination is a directory, copy file into it.
            # If destination is a file path, copy and potentially rename.
            # If destination exists and is a file, shutil.copy2 will overwrite it.
            # If destination exists and is a dir, this will error; handle by forming full path.
            if destination_path.is_dir():
                final_dest_path = destination_path / source_path.name
            else:
                final_dest_path = destination_path

            if final_dest_path.exists() and not final_dest_path.is_dir():
                 logger.warning(f"Destination file '{final_dest_path}' exists and will be overwritten.")
            elif final_dest_path.is_dir(): # Should not happen if source is file and dest is dir due to above
                 return {"error": f"Cannot overwrite directory '{final_dest_path}' with a file."}


            shutil.copy2(str(source_path), str(final_dest_path)) # copy2 preserves metadata
            logger.info(f"Successfully copied file '{source_path}' to '{final_dest_path}'.")
            return {
                "source_path": str(source_path),
                "destination_path": str(final_dest_path),
                "message": f"File copied from '{source_path}' to '{final_dest_path}'.",
            }
        elif source_path.is_dir():
            # For directories, destination_path must be the new directory's path (or parent if name is same)
            # shutil.copytree expects the destination directory not to exist or be empty for some versions.
            # A common pattern is to copy into a subdirectory of destination_path if destination_path itself is an existing dir.
            final_dest_dir = destination_path
            if destination_path.exists():
                if not destination_path.is_dir():
                     return {"error": f"Cannot copy directory to a path that is an existing file: {destination_path}"}
                # If destination is an existing directory, copy the source directory *into* it
                final_dest_dir = destination_path / source_path.name
                if final_dest_dir.exists():
                    return {"error": f"Destination directory '{final_dest_dir}' already exists. Cannot overwrite directory with copytree."}

            shutil.copytree(str(source_path), str(final_dest_dir))
            logger.info(f"Successfully copied directory '{source_path}' to '{final_dest_dir}'.")
            return {
                "source_path": str(source_path),
                "destination_path": str(final_dest_dir),
                "message": f"Directory copied from '{source_path}' to '{final_dest_dir}'.",
            }
        else:
            return {"error": f"Source path '{source_path}' is neither a file nor a directory."}

    except PermissionError:
        logger.error(f"Permission denied for copying '{source_path_str}' to '{destination_path_str}'")
        return {"error": f"Permission denied for copying operation."}
    except FileExistsError as fee: # Specifically for copytree if dest exists and is not empty
        logger.error(f"FileExistsError during copy: {fee}", exc_info=True)
        return {"error": str(fee)}
    except Exception as e:
        logger.error(f"Error copying '{source_path_str}' to '{destination_path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while copying: {e}"}


def get_file_properties(path_str: str) -> Dict[str, Union[str, int, float, None]]:
    """
    Gets properties of a file or directory.

    Args:
        path_str: The path to the file or directory.

    Returns:
        A dictionary with properties like "name", "path", "type", "size_bytes",
        "modification_time", "creation_time", "is_readable", "is_writable",
        or "error" if the path doesn't exist or an error occurs.
    """
    try:
        path = Path(path_str).resolve()
        if not path.exists():
            return {"error": f"Path '{path}' does not exist."}

        stat_info = path.stat()
        import os
        import datetime

        properties = {
            "name": path.name,
            "path": str(path),
            "type": "directory" if path.is_dir() else "file" if path.is_file() else "other",
            "size_bytes": stat_info.st_size,
            "modification_time_iso": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "creation_time_iso": datetime.datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "access_time_iso": datetime.datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            "is_readable": os.access(path, os.R_OK),
            "is_writable": os.access(path, os.W_OK),
            "is_executable": os.access(path, os.X_OK) if path.is_file() else None, # Executable check more relevant for files
        }
        logger.info(f"Retrieved properties for '{path}'.")
        return properties
    except PermissionError:
        logger.error(f"Permission denied for getting properties of: {path_str}")
        return {"error": f"Permission denied for getting properties of: {path_str}"}
    except Exception as e:
        logger.error(f"Error getting properties for '{path_str}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while getting properties: {e}"}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    logger.info("File System Module Example - Extended Tests")

    # --- Test Directory Setup ---
    base_test_dir = Path("./__test_fs_operations__")
    base_test_dir.mkdir(exist_ok=True)

    dir_to_create = base_test_dir / "newly_created_dir"
    file_to_write = base_test_dir / "test_write_file.txt"
    file_to_append = base_test_dir / "test_append_file.txt"
    original_for_move = base_test_dir / "original_for_move.txt"
    move_dest_dir = base_test_dir / "move_destination_dir"
    move_dest_file = move_dest_dir / original_for_move.name
    original_for_copy_file = base_test_dir / "original_for_copy.txt"
    copy_dest_file = base_test_dir / "copied_file.txt"
    original_for_copy_dir = base_test_dir / "original_dir_for_copy"
    original_for_copy_dir_subfile = original_for_copy_dir / "subfile.txt"
    copy_dest_dir = base_test_dir / "copied_dir"


    def cleanup_test_environment():
        logger.info("--- Cleaning up test environment ---")
        import shutil
        if file_to_write.exists(): file_to_write.unlink()
        if file_to_append.exists(): file_to_append.unlink()
        if original_for_move.exists(): original_for_move.unlink()
        if move_dest_file.exists(): move_dest_file.unlink()
        if move_dest_dir.exists(): shutil.rmtree(move_dest_dir, ignore_errors=True)
        if original_for_copy_file.exists(): original_for_copy_file.unlink()
        if copy_dest_file.exists(): copy_dest_file.unlink()
        if original_for_copy_dir.exists(): shutil.rmtree(original_for_copy_dir, ignore_errors=True)
        if copy_dest_dir.exists(): shutil.rmtree(copy_dest_dir, ignore_errors=True)
        if dir_to_create.exists(): shutil.rmtree(dir_to_create, ignore_errors=True)
        if base_test_dir.exists(): shutil.rmtree(base_test_dir, ignore_errors=True)
        logger.info("Test environment cleanup complete.")

    try:
        # --- Test get_file_properties (on existing test_fs_dir from previous example if needed) ---
        # Assuming test_fs_dir might still exist from a prior run or be created by other tests
        logger.info(f"\n--- Testing get_file_properties on script itself: {__file__} ---")
        props = get_file_properties(__file__)
        if "error" in props: print(f"Error: {props['error']}")
        else: print(f"Properties for {props.get('path')}: Size: {props.get('size_bytes')}B, Type: {props.get('type')}, Modified: {props.get('modification_time_iso')}")

        # --- Test create_directory ---
        logger.info(f"\n--- Testing create_directory: {dir_to_create} ---")
        result = create_directory(str(dir_to_create))
        print(result)
        assert dir_to_create.is_dir(), "Directory creation failed"
        result_exists = create_directory(str(dir_to_create)) # Test creating if exists
        print(f"Attempting to create existing directory: {result_exists}")
        assert "already exists" in result_exists.get("message", ""), "Creating existing directory should report so"


        # --- Test write_text_file ---
        logger.info(f"\n--- Testing write_text_file: {file_to_write} ---")
        content_to_write = "Hello from write_text_file!\nThis is a new line."
        result = write_text_file(str(file_to_write), content_to_write)
        print(result)
        assert file_to_write.is_file() and file_to_write.read_text() == content_to_write, "Write file failed"

        logger.info(f"\n--- Testing write_text_file (no overwrite): {file_to_write} ---")
        result_no_overwrite = write_text_file(str(file_to_write), "Attempt to overwrite.", overwrite=False)
        print(result_no_overwrite)
        assert "error" in result_no_overwrite and "already exists" in result_no_overwrite["error"], "Should not overwrite by default"
        assert file_to_write.read_text() == content_to_write, "File content should not have changed"

        logger.info(f"\n--- Testing write_text_file (with overwrite): {file_to_write} ---")
        new_content = "Overwritten content."
        result_overwrite = write_text_file(str(file_to_write), new_content, overwrite=True)
        print(result_overwrite)
        assert file_to_write.read_text() == new_content, "File overwrite failed"

        # --- Test append_text_to_file ---
        logger.info(f"\n--- Testing append_text_to_file (new file): {file_to_append} ---")
        initial_append_content = "Initial content for append.\n"
        result = append_text_to_file(str(file_to_append), initial_append_content)
        print(result)
        assert file_to_append.is_file() and file_to_append.read_text() == initial_append_content, "Append to new file failed"

        logger.info(f"\n--- Testing append_text_to_file (existing file): {file_to_append} ---")
        additional_append_content = "Appended line."
        result = append_text_to_file(str(file_to_append), additional_append_content)
        print(result)
        expected_appended_content = initial_append_content + additional_append_content
        assert file_to_append.read_text() == expected_appended_content, "Append to existing file failed"

        # --- Test copy_file_or_directory (file) ---
        logger.info(f"\n--- Testing copy_file_or_directory (file): {original_for_copy_file} to {copy_dest_file} ---")
        original_for_copy_file.write_text("Content to be copied.")
        result = copy_file_or_directory(str(original_for_copy_file), str(copy_dest_file))
        print(result)
        assert copy_dest_file.is_file() and copy_dest_file.read_text() == original_for_copy_file.read_text(), "Copy file failed"

        # --- Test copy_file_or_directory (directory) ---
        logger.info(f"\n--- Testing copy_file_or_directory (directory): {original_for_copy_dir} to {copy_dest_dir} ---")
        original_for_copy_dir.mkdir(exist_ok=True)
        original_for_copy_dir_subfile.write_text("Subfile content in dir to copy.")
        result = copy_file_or_directory(str(original_for_copy_dir), str(copy_dest_dir))
        print(result)
        assert copy_dest_dir.is_dir(), "Copy directory failed - dest dir not created"
        assert (copy_dest_dir / original_for_copy_dir_subfile.name).is_file(), "Copy directory failed - subfile not copied"
        assert (copy_dest_dir / original_for_copy_dir_subfile.name).read_text() == original_for_copy_dir_subfile.read_text(), "Copied subfile content mismatch"

        logger.info(f"\n--- Testing copy_file_or_directory (directory into existing dir) ---")
        copy_into_existing_dir_target = base_test_dir / "existing_target_for_copy"
        copy_into_existing_dir_target.mkdir(exist_ok=True)
        # We expect original_for_copy_dir to be copied *inside* existing_target_for_copy
        expected_final_copied_dir_path = copy_into_existing_dir_target / original_for_copy_dir.name
        result_copy_into = copy_file_or_directory(str(original_for_copy_dir), str(copy_into_existing_dir_target))
        print(result_copy_into)
        assert expected_final_copied_dir_path.is_dir(), "Copy directory into existing directory failed"
        assert (expected_final_copied_dir_path / original_for_copy_dir_subfile.name).is_file(), "Subfile not copied when copying dir into existing dir"


        # --- Test move_file_or_directory (file) ---
        logger.info(f"\n--- Testing move_file_or_directory (file): {original_for_move} to {move_dest_file} ---")
        original_for_move.write_text("Content to be moved.")
        move_dest_dir.mkdir(exist_ok=True) # Ensure destination directory exists
        result = move_file_or_directory(str(original_for_move), str(move_dest_file))
        print(result)
        assert not original_for_move.exists(), "Move file failed - source still exists"
        assert move_dest_file.is_file() and move_dest_file.read_text() == "Content to be moved.", "Move file failed - content mismatch or not found at dest"

        # --- Test delete_file_or_directory (file) ---
        # Recreate file to delete it
        file_to_delete_explicitly = base_test_dir / "explicit_delete_me.txt"
        file_to_delete_explicitly.write_text("Delete this content.")
        logger.info(f"\n--- Testing delete_file_or_directory (file): {file_to_delete_explicitly} ---")
        result = delete_file_or_directory(str(file_to_delete_explicitly))
        print(result)
        assert not file_to_delete_explicitly.exists(), "Delete file failed"

        # --- Test delete_file_or_directory (directory) ---
        dir_to_delete_explicitly = base_test_dir / "explicit_delete_me_dir"
        dir_to_delete_explicitly.mkdir(exist_ok=True)
        (dir_to_delete_explicitly / "somefile.txt").write_text("Inside dir to delete.")
        logger.info(f"\n--- Testing delete_file_or_directory (directory): {dir_to_delete_explicitly} ---")
        result = delete_file_or_directory(str(dir_to_delete_explicitly))
        print(result)
        assert not dir_to_delete_explicitly.exists(), "Delete directory failed"

        logger.info("\n--- All new file system operation tests completed (if no asserts failed) ---")

    except AssertionError as ae:
        logger.error(f"TEST ASSERTION FAILED: {ae}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred during file system tests: {e}", exc_info=True)
    finally:
        cleanup_test_environment()

    logger.info("File system module example and extended tests finished.")
