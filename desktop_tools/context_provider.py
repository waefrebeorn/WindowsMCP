import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

try:
    from .context_indexer import ContextIndexer, FileNode # Assuming relative import works
except ImportError:
    # Fallback for direct execution or different project structures
    from context_indexer import ContextIndexer, FileNode


class ContextProvider:
    """
    Gathers and provides context from a project, including file contents,
    editor state (simulated), and user query analysis for @-references.
    """
    def __init__(self, project_root_str: str):
        self.project_root = Path(project_root_str).resolve()
        self.indexer = ContextIndexer(project_root_str)
        self.indexer.index_project() # Initial indexing

        self.current_file_path: Optional[Path] = None
        self.cursor_position: Optional[Tuple[int, int]] = None # (line, column)
        self.open_file_paths: List[Path] = []

    def update_editor_state(self,
                            current_file_rel_path: Optional[str] = None,
                            cursor_pos: Optional[Tuple[int, int]] = None,
                            open_files_rel_paths: Optional[List[str]] = None):
        """
        Updates the simulated state of the editor.
        Paths should be relative to the project root.
        """
        if current_file_rel_path:
            self.current_file_path = (self.project_root / current_file_rel_path).resolve()
            # Basic validation: check if it's part of indexed files (implies it's not ignored)
            if self.current_file_path not in self.indexer.file_hashes:
                print(f"Warning: Current file '{current_file_rel_path}' is not in the project index or is ignored.")
                self.current_file_path = None # Invalidate if not found/ignored
        else:
            self.current_file_path = None

        self.cursor_position = cursor_pos

        self.open_file_paths = []
        if open_files_rel_paths:
            for rel_path in open_files_rel_paths:
                abs_path = (self.project_root / rel_path).resolve()
                # Basic validation
                if abs_path in self.indexer.file_hashes:
                    self.open_file_paths.append(abs_path)
                else:
                    print(f"Warning: Open file '{rel_path}' is not in the project index or is ignored. Skipping.")

        # Ensure current file is part of open files if both are set
        if self.current_file_path and self.current_file_path not in self.open_file_paths:
             if self.current_file_path in self.indexer.file_hashes: # Check again if it's valid before adding
                self.open_file_paths.append(self.current_file_path)


    def _get_file_content(self, file_path: Path, max_chars: Optional[int] = None) -> Optional[str]:
        """
        Reads the content of a file. Optionally truncates to max_chars.
        Returns None if file cannot be read or is not in index.
        """
        if not file_path.is_absolute():
            file_path = (self.project_root / file_path).resolve()

        if file_path not in self.indexer.file_hashes:
            # This check also implicitly ensures the file is not ignored by the indexer
            print(f"Warning: File '{file_path.relative_to(self.project_root)}' not found in index or is ignored.")
            return None

        try:
            content = file_path.read_text(encoding='utf-8')
            if max_chars is not None and len(content) > max_chars:
                return content[:max_chars] + "\n... [truncated]"
            return content
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        except Exception as e: # Catch other potential errors like decoding errors
            print(f"An unexpected error occurred while reading file {file_path}: {e}")
            return None

    def _extract_at_references(self, text: str) -> List[Path]:
        """
        Extracts @-referenced file paths from text.
        Currently supports basic @path/to/file.ext patterns.
        """
        # Regex to find @ followed by non-whitespace characters that typically form a path
        # This is a simplified regex and might need refinement for complex paths or quoted paths.
        # It looks for @ followed by characters suitable for file paths, excluding spaces unless quoted.
        # For simplicity now: @ followed by non-space chars.
        matches = re.findall(r'@([\S]+)', text)
        referenced_paths: List[Path] = []

        for rel_path_str in matches:
            # Normalize potential path issues, e.g. if it includes trailing punctuation by mistake
            # This is a simple heuristic; more robust parsing might be needed.
            # For now, assume the matched string is a direct relative path.
            # Example: @src/main.py, or @file.txt

            # Remove common trailing punctuation that might be accidentally included
            rel_path_str = rel_path_str.rstrip('.,;:!?')

            abs_path = (self.project_root / rel_path_str).resolve()

            # Validate if this path exists in our indexed files
            if abs_path in self.indexer.file_hashes:
                referenced_paths.append(abs_path)
            else:
                # Try to see if it's a partial match or needs .py/.js etc. (more advanced)
                # For now, just warn if not found directly.
                print(f"Warning: Referenced file '@{rel_path_str}' not found in project index or is ignored.")

        return list(set(referenced_paths)) # Return unique paths

    def gather_context(self, user_query: str, max_file_chars_snippet: int = 500, max_file_chars_full: int = 2000) -> Dict[str, Any]:
        """
        Assembles the context based on the current state and user query.
        """
        context_data: Dict[str, Any] = {
            "project_root": str(self.project_root),
            "project_root_hash": self.indexer.get_root_hash(),
            "user_query": user_query,
            "current_file": None,
            "cursor_position": self.cursor_position,
            "open_files": [],
            "referenced_files": []
        }

        # 1. Current file context
        if self.current_file_path:
            content = self._get_file_content(self.current_file_path, max_chars=max_file_chars_full)
            if content:
                context_data["current_file"] = {
                    "path": str(self.current_file_path.relative_to(self.project_root)),
                    "content": content,
                    "hash": self.indexer.get_file_hash(self.current_file_path),
                    "cursor_snippet": None # Initialize
                }

                if self.cursor_position and content:
                    try:
                        # Assuming cursor_position is (1-indexed line, 1-indexed column)
                        cursor_line_1_indexed = self.cursor_position[0]
                        # cursor_col_1_indexed = self.cursor_position[1] # Column not used for line snippet

                        if cursor_line_1_indexed > 0:
                            lines = content.splitlines()
                            cursor_line_0_indexed = cursor_line_1_indexed - 1

                            if 0 <= cursor_line_0_indexed < len(lines):
                                lines_around_cursor = 5 # Number of lines above and below
                                snippet_start_line = max(0, cursor_line_0_indexed - lines_around_cursor)
                                snippet_end_line = min(len(lines), cursor_line_0_indexed + lines_around_cursor + 1)

                                snippet_lines = lines[snippet_start_line:snippet_end_line]
                                # Add a marker for the cursor line if desired, e.g., "> "
                                # For simplicity, just join the lines for now.
                                # Could also include line numbers.
                                context_data["current_file"]["cursor_snippet"] = "\n".join(snippet_lines)
                                # Add a pointer to the exact line in the snippet
                                if context_data["current_file"]["cursor_snippet"]:
                                     snippet_lines_with_marker = []
                                     for i, line_content in enumerate(snippet_lines):
                                         actual_line_num_0_indexed = snippet_start_line + i
                                         prefix = "> " if actual_line_num_0_indexed == cursor_line_0_indexed else "  "
                                         snippet_lines_with_marker.append(f"{prefix}{line_content}")
                                     context_data["current_file"]["cursor_snippet_formatted"] = "\n".join(snippet_lines_with_marker)


                            else:
                                print(f"Warning: Cursor line {cursor_line_1_indexed} is out of bounds for file {context_data['current_file']['path']} (total lines: {len(lines)}).")
                        else:
                             print(f"Warning: Invalid cursor line number {cursor_line_1_indexed}.")
                    except Exception as e:
                        print(f"Error generating cursor snippet: {e}")


        # 2. Open files context (excluding current file if already added)
        for f_path in self.open_file_paths:
            if f_path == self.current_file_path and context_data["current_file"] is not None:
                continue # Already processed as current_file

            content_snippet = self._get_file_content(f_path, max_chars=max_file_chars_snippet)
            if content_snippet:
                context_data["open_files"].append({
                    "path": str(f_path.relative_to(self.project_root)),
                    "content_snippet": content_snippet,
                    "hash": self.indexer.get_file_hash(f_path)
                })

        # 3. @-referenced files from user query
        at_referenced_paths = self._extract_at_references(user_query)
        for ref_path in at_referenced_paths:
            # Avoid duplicating if already included as current or open file (full content preferred)
            is_current = (ref_path == self.current_file_path and context_data["current_file"] is not None)
            is_open = any(entry["path"] == str(ref_path.relative_to(self.project_root)) for entry in context_data["open_files"])

            if is_current or is_open:
                # If it's the current file, its full content is already there.
                # If it's an open file, its snippet is already there.
                # We could choose to upgrade an open file's snippet to full content if referenced.
                # For now, just ensure it's noted or skip if already well-represented.
                # Let's ensure it's at least mentioned as "referenced" even if content is elsewhere.
                # This part can be refined based on how LLM should prioritize.
                pass # Already handled or its snippet is present

            content_full = self._get_file_content(ref_path, max_chars=max_file_chars_full)
            if content_full:
                context_data["referenced_files"].append({
                    "path": str(ref_path.relative_to(self.project_root)),
                    "content": content_full, # Provide more content for @-refs
                    "hash": self.indexer.get_file_hash(ref_path)
                })

        # Remove duplicates between open_files and referenced_files if any path is same
        # Favor the one from referenced_files if it has more content
        open_files_paths_set = {Path(self.project_root / entry["path"]) for entry in context_data["open_files"]}
        final_referenced_files = []
        for ref_entry in context_data["referenced_files"]:
            ref_path_abs = self.project_root / ref_entry["path"]
            if ref_path_abs not in open_files_paths_set:
                final_referenced_files.append(ref_entry)
            else: # It was also in open_files, check if current ref_entry is better
                # This logic assumes referenced_files gets full content, open_files gets snippets
                # So, if a file is both open and @-referenced, the @-reference (fuller content) takes precedence.
                # We need to remove the snippet version from open_files.
                context_data["open_files"] = [
                    of for of in context_data["open_files"]
                    if Path(self.project_root / of["path"]) != ref_path_abs
                ]
                final_referenced_files.append(ref_entry)
        context_data["referenced_files"] = final_referenced_files


        return context_data

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Assuming dummy_wubu_project from context_indexer.py example still exists or can be recreated
    dummy_project_path_str = "./dummy_wubu_project"
    dummy_project_path = Path(dummy_project_path_str)

    if not dummy_project_path.exists():
        print(f"Dummy project path {dummy_project_path_str} not found. Please run context_indexer.py first to create it.")
    else:
        provider = ContextProvider(dummy_project_path_str)

        # Simulate editor state
        provider.update_editor_state(
            current_file_rel_path="file1.txt",
            cursor_pos=(1, 5),
            open_files_rel_paths=["file1.txt", "file2.py", "non_existent_file.txt"] # non_existent should be skipped
        )

        print("\nInitial state after update_editor_state:")
        print(f"Current file: {provider.current_file_path}")
        print(f"Open files: {provider.open_file_paths}")


        # Test @-reference extraction
        print("\nTesting @-reference extraction:")
        test_query_at_refs = "Can you look at @file1.txt and also @file2.py? What about @ignored_dir/secret.txt and @non_existent.py?"
        extracted = provider._extract_at_references(test_query_at_refs)
        print(f"From query: '{test_query_at_refs}'")
        print(f"Extracted and valid paths: {[str(p.relative_to(provider.project_root)) for p in extracted]}")

        # Test context gathering
        print("\nGathering context for query: " + test_query_at_refs)
        context = provider.gather_context(test_query_at_refs)

        print("\n--- Assembled Context ---")
        print(f"Project Root: {context['project_root']}")
        print(f"Project Root Hash: {context['project_root_hash']}")
        print(f"User Query: {context['user_query']}")

        if context['current_file']:
            print("\nCurrent File:")
            print(f"  Path: {context['current_file']['path']}")
            print(f"  Hash: {context['current_file']['hash']}")
            print(f"  Content: \n{context['current_file']['content'][:100]}...\n")
            if context['current_file'].get('cursor_snippet_formatted'):
                print(f"  Cursor Snippet (formatted):\n{context['current_file']['cursor_snippet_formatted']}\n")
            elif context['current_file'].get('cursor_snippet'):
                print(f"  Cursor Snippet (raw):\n{context['current_file']['cursor_snippet']}\n")


        print("Cursor Position:", context['cursor_position'])

        if context['open_files']:
            print("\nOpen Files (snippets):")
            for f_info in context['open_files']:
                print(f"  Path: {f_info['path']}")
                print(f"  Hash: {f_info['hash']}")
                print(f"  Content Snippet: \n{f_info['content_snippet'][:100]}...\n")

        if context['referenced_files']:
            print("\nReferenced Files (full content from @):")
            for f_info in context['referenced_files']:
                print(f"  Path: {f_info['path']}")
                print(f"  Hash: {f_info['hash']}")
                print(f"  Content: \n{f_info['content'][:100]}...\n")
        print("--- End of Context ---")

        # Test with a file that should be ignored by .gitignore
        provider.update_editor_state(current_file_rel_path="data.log") # data.log is in .gitignore
        print(f"\nTrying to set current file to ignored 'data.log': {provider.current_file_path}") # Should be None or print warning

        context_ignored_query = "What about @data.log?"
        print(f"\nGathering context for query: {context_ignored_query}")
        context_ignored = provider.gather_context(context_ignored_query)
        print(f"Referenced files for ignored query: {context_ignored['referenced_files']}") # Should be empty for data.log
```
