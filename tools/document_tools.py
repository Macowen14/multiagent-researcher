import os
import logging
from typing import List, Annotated, Optional, Dict
from langchain_core.tools import tool
from utils.util import make_file_path, safe_write_to_file, acquire_file_lock

logger = logging.getLogger(__name__)

@tool('create_outline', description="Creates an outline from a list of points and saves it to a specified file.")
def create_outline(
    points: Annotated[List[str], "List of points to include in the outline"],
    filename: Annotated[str, "Name of the file to save the outline to"]
) -> Annotated[str, "Confirmation message that the outline was created and saved and in what file"]:
    """Creates an outline from a list of points and saves it to a specified file."""
    logger.info(f"Creating outline with points: {points} and saving to {filename}")
    outline = "\n".join(f"- {point}" for point in points)
    
    # Use our combined safe function
    return safe_write_to_file(filename, outline)

@tool("read_document")
def read_document(
    file_name: Annotated[str, "File path to read the document from"],
    start: Annotated[Optional[int], "The start line, Default is 0"] = None,
    end: Annotated[Optional[int], "The end of the line. Default is none"] = None
) -> str:
    """Read the specified document contents. Creates file with intelligent name if it doesn't exist."""
    file_to_use = make_file_path(file_name)
    if not os.path.exists(file_to_use):
        # Create intelligent filename if file doesn't exist
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract meaningful name from requested filename
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        if not base_name or base_name == file_name:
            base_name = "untitled"
        
        # Create new filename with timestamp
        new_filename = f"{base_name}_created_{timestamp}.md"
        new_filepath = make_file_path(new_filename)
        
        # Create the file with a header
        header = f"# {base_name.replace('_', ' ').title()}\n\n*Created automatically on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        safe_write_to_file(new_filename, header)
        
        return f"File '{file_name}' did not exist. Created new file '{new_filename}' with intelligent naming."
        
    try:
        # We grab the lock even when reading so we don't read a half-written file!
        with acquire_file_lock(file_to_use, timeout=5):
            with open(file_to_use, "r") as f:
                lines = f.readlines()
    except TimeoutError:
        return f"Error: File {file_name} is currently locked by another process."
        
    if start is None:
        start = 0
    return "".join(lines[start:end])

@tool("write_document")
def write_document(
    content: Annotated[str, "Text content to be written to the document"],
    filename: Annotated[str, "File path to save the document"]
) -> str:
    """Create and save a text document"""
    # Use our combined safe function
    return safe_write_to_file(filename, content)

@tool("edit_document")
def edit_document(
    filename: Annotated[str, "File path to save the document"],
    inserts: Annotated[Dict[int, str], "Dictionary where key is the line number and the value is the text to be inserted at the line"]
) -> str:
    """Edit a document by inserting text at specified line numbers. Creates file if it doesn't exist."""
    file_to_use = make_file_path(filename)
    if not os.path.exists(file_to_use):
        # Create the file first if it doesn't exist
        import datetime
        base_name = os.path.splitext(os.path.basename(filename))[0]
        header = f"# {base_name.replace('_', ' ').title()}\n\n*Created automatically on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        safe_write_to_file(filename, header)
        
    try:
        # Wrap both the read and the write inside the lock
        with acquire_file_lock(file_to_use, timeout=10):
            with open(file_to_use, 'r') as f:
                lines = f.readlines()

            # Sort inserts in descending order so earlier insertions don't offset the indices for later ones
            sorted_inserts = sorted(inserts.items(), reverse=True)

            for line_number, text in sorted_inserts:
                if 1 <= line_number <= len(lines) + 1:
                    # line_number is 1-indexed
                    lines.insert(line_number - 1, text + "\n")
                else:
                    return f"Error: line number {line_number} is out of range"

            with open(file_to_use, 'w') as f:
                f.writelines(lines)

        return f"Document edited and saved to {filename}"
    except TimeoutError:
        return f"Error: The file {filename} is currently locked by another process."
