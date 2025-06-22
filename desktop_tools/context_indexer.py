import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import fnmatch

class FileNode:
    """Represents a file with its path and hash."""
    def __init__(self, path: Path, hash_value: str):
        self.path = path
        self.hash = hash_value

    def __repr__(self):
        return f"FileNode(path='{self.path}', hash='{self.hash[:7]}...')"

class MerkleNode:
    """Represents a node in the Merkle tree."""
    def __init__(self, hash_value: str):
        self.hash: str = hash_value
        self.left: Optional[MerkleNode] = None
        self.right: Optional[MerkleNode] = None
        # If it's a leaf node directly representing a file, we can store the FileNode
        self.file_node: Optional[FileNode] = None

    def __repr__(self):
        return f"MerkleNode(hash='{self.hash[:7]}...', file='{self.file_node.path.name if self.file_node else None}')"

def hash_string(data: str) -> str:
    """Hashes a string using SHA256."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def hash_file_content(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file's content."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in 64k chunks
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except IOError:
        # Handle cases where file might be inaccessible during hashing
        return "" # Return empty hash or raise specific error

class ContextIndexer:
    """
    Indexes a project directory, builds a Merkle tree of file hashes,
    and handles ignore patterns.
    """
    def __init__(self, project_root_str: str):
        self.project_root: Path = Path(project_root_str).resolve()
        self.merkle_root: Optional[MerkleNode] = None
        self.file_hashes: Dict[Path, str] = {} # Stores individual file hashes path -> hash
        self.ignore_patterns: List[str] = []
        self.WUBU_IGNORE_FILE = ".wubuignore" # Or .cursorignore as per original doc

        if not self.project_root.is_dir():
            raise ValueError(f"Project root not found or is not a directory: {self.project_root}")

    def _load_ignore_patterns(self):
        """Loads patterns from .gitignore and .wubuignore files."""
        self.ignore_patterns = []
        for ignore_file_name in [".gitignore", self.WUBU_IGNORE_FILE]:
            ignore_file_path = self.project_root / ignore_file_name
            if ignore_file_path.is_file():
                try:
                    with open(ignore_file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                self.ignore_patterns.append(line)
                except IOError as e:
                    print(f"Warning: Could not read ignore file {ignore_file_path}: {e}")
        # Add common SCM and editor directories by default if not covered
        # Example: add common patterns like '.git/', '.vscode/', '__pycache__/'
        default_ignores = ['.git/', '.svn/', '.hg/', '.vscode/', '__pycache__/', '*.pyc', '*.pyo', '.DS_Store']
        for pattern in default_ignores:
            if pattern not in self.ignore_patterns: # Avoid duplicates if user already has them
                self.ignore_patterns.append(pattern)


    def _should_ignore(self, entry_path: Path) -> bool:
        """
        Checks if a file or directory path should be ignored.
        Compares the path relative to the project root against ignore patterns.
        """
        relative_path_str = str(entry_path.relative_to(self.project_root))
        # For directories, ensure patterns like 'node_modules/' match 'node_modules'
        path_to_check_str = relative_path_str
        if entry_path.is_dir() and not relative_path_str.endswith('/'):
            path_to_check_str += '/'

        for pattern in self.ignore_patterns:
            # Normalize pattern: remove leading/trailing slashes for directory matching simplicity with fnmatch
            # More robust gitignore parsing would be needed for complex rules (e.g. `!pattern`, `**/pattern`)
            normalized_pattern = pattern.strip('/')

            # Direct match for files or directory names
            if fnmatch.fnmatch(relative_path_str, pattern): # e.g. file.txt, dir/file.txt
                return True
            if fnmatch.fnmatch(path_to_check_str, pattern): # e.g. dir/, dir/subdir/
                 return True
            # Match if pattern is a prefix of the path (for directory patterns like 'build/')
            if pattern.endswith('/') and path_to_check_str.startswith(pattern):
                return True
            # Match for patterns like '*.log' or 'temp*'
            if '*' in pattern or '?' in pattern or '[' in pattern:
                 if fnmatch.fnmatch(entry_path.name, pattern): # Match against file/dir name
                    return True
                 # Check if any part of the path matches for patterns like '*/temp/*'
                 # This part needs careful implementation for full gitignore compatibility
                 # For now, we rely on fnmatch on the relative path.

        return False

    def _build_merkle_tree_recursive(self, nodes: List[MerkleNode]) -> Optional[MerkleNode]:
        """Recursively builds the Merkle tree."""
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        new_level_nodes: List[MerkleNode] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1] if (i+1) < len(nodes) else left # Duplicate last if odd number

            combined_hash_data = left.hash + right.hash
            parent_hash = hash_string(combined_hash_data)

            parent_node = MerkleNode(parent_hash)
            parent_node.left = left
            parent_node.right = right
            new_level_nodes.append(parent_node)

        return self._build_merkle_tree_recursive(new_level_nodes)

    def index_project(self):
        """
        Indexes all files in the project directory, calculates their hashes,
        and builds a Merkle tree.
        """
        self.file_hashes.clear()
        self._load_ignore_patterns()

        file_nodes: List[FileNode] = []

        for root, dirs, files in os.walk(self.project_root, topdown=True):
            current_path = Path(root)

            # Filter directories based on ignore patterns
            # os.walk allows modifying dirs in-place to prune search
            dirs[:] = [d for d in dirs if not self._should_ignore(current_path / d)]

            for name in files:
                file_path = current_path / name
                if not self._should_ignore(file_path):
                    file_hash = hash_file_content(file_path)
                    if file_hash: # Only include files that could be hashed
                        self.file_hashes[file_path] = file_hash
                        file_nodes.append(FileNode(file_path, file_hash))

        # Sort file_nodes by path to ensure consistent tree structure
        file_nodes.sort(key=lambda fn: fn.path)

        if not file_nodes:
            self.merkle_root = None
            print("No files found to index.")
            return

        # Create leaf MerkleNodes from FileNodes
        leaf_merkle_nodes: List[MerkleNode] = []
        for fn in file_nodes:
            # Hash of a leaf MerkleNode can be the file hash itself, or hash of (path+hash)
            # For simplicity, let's use file hash.
            leaf_node = MerkleNode(fn.hash)
            leaf_node.file_node = fn
            leaf_merkle_nodes.append(leaf_node)

        self.merkle_root = self._build_merkle_tree_recursive(leaf_merkle_nodes)
        if self.merkle_root:
            print(f"Project indexed. Merkle root: {self.merkle_root.hash[:10]}... with {len(file_nodes)} files.")
        else:
            print("Failed to build Merkle tree.")

    def get_root_hash(self) -> Optional[str]:
        """Returns the hash of the Merkle tree root."""
        return self.merkle_root.hash if self.merkle_root else None

    def get_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """Returns the stored hash for a given file path."""
        p = Path(file_path)
        if not p.is_absolute():
            p = (self.project_root / p).resolve()
        return self.file_hashes.get(p)

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Create a dummy project structure for testing
    dummy_project_path = Path("./dummy_wubu_project")
    dummy_project_path.mkdir(exist_ok=True)
    (dummy_project_path / "file1.txt").write_text("Hello WuBu")
    (dummy_project_path / "file2.py").write_text("print('WuBu')")
    (dummy_project_path / ".gitignore").write_text("*.log\nignored_dir/\n.idea/")
    (dummy_project_path / ".wubuignore").write_text("*.tmp\n")

    ignored_dir = dummy_project_path / "ignored_dir"
    ignored_dir.mkdir(exist_ok=True)
    (ignored_dir / "secret.txt").write_text("secret content")
    (dummy_project_path / "data.log").write_text("log data")
    idea_dir = dummy_project_path / ".idea"
    idea_dir.mkdir(exist_ok=True)
    (idea_dir / "workspace.xml").write_text("<xml></xml>")


    print(f"Indexing dummy project at: {dummy_project_path.resolve()}")
    indexer = ContextIndexer(str(dummy_project_path.resolve()))
    indexer.index_project()

    if indexer.merkle_root:
        print(f"Merkle Root Hash: {indexer.get_root_hash()}")
        print("Indexed file hashes:")
        for f_path, f_hash in indexer.file_hashes.items():
            print(f"  {f_path.relative_to(dummy_project_path)}: {f_hash[:10]}...")

    print("\nTesting _should_ignore:")
    print(f"Should ignore 'data.log': {indexer._should_ignore(dummy_project_path / 'data.log')}") # True
    print(f"Should ignore 'ignored_dir/secret.txt': {indexer._should_ignore(dummy_project_path / 'ignored_dir' / 'secret.txt')}") # True (due to dir)
    print(f"Should ignore '.idea/workspace.xml': {indexer._should_ignore(dummy_project_path / '.idea' / 'workspace.xml')}") # True
    print(f"Should ignore 'file1.txt': {indexer._should_ignore(dummy_project_path / 'file1.txt')}") # False

    # Clean up dummy project (optional)
    # import shutil
    # shutil.rmtree(dummy_project_path)
    # print(f"\nCleaned up {dummy_project_path}")
