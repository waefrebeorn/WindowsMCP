import ast
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CodeStructureVisitor(ast.NodeVisitor):
    """
    An AST visitor to extract information about functions and classes.
    """
    def __init__(self):
        self.structures: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extracts info for top-level functions and methods."""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else start_line

        # Get decorator names if any
        decorator_list_str = [ast.unparse(d) if hasattr(ast, 'unparse') else d.id for d in node.decorator_list if hasattr(d, 'id') or hasattr(ast, 'unparse')]


        self.structures.append({
            "type": "function",
            "name": node.name,
            "start_line": start_line,
            "end_line": end_line,
            "args": [arg.arg for arg in node.args.args],
            "decorators": decorator_list_str,
            "docstring": ast.get_docstring(node)
        })
        self.generic_visit(node) # Visit children (e.g., nested functions, though less common for detailed outline)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extracts info for async functions and methods."""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else start_line
        decorator_list_str = [ast.unparse(d) if hasattr(ast, 'unparse') else d.id for d in node.decorator_list if hasattr(d, 'id') or hasattr(ast, 'unparse')]

        self.structures.append({
            "type": "async_function",
            "name": node.name,
            "start_line": start_line,
            "end_line": end_line,
            "args": [arg.arg for arg in node.args.args],
            "decorators": decorator_list_str,
            "docstring": ast.get_docstring(node)
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extracts info for classes."""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else start_line
        decorator_list_str = [ast.unparse(d) if hasattr(ast, 'unparse') else d.id for d in node.decorator_list if hasattr(d, 'id') or hasattr(ast, 'unparse')]


        class_info: Dict[str, Any] = {
            "type": "class",
            "name": node.name,
            "start_line": start_line,
            "end_line": end_line,
            "methods": [],
            "nested_classes": [], # For completeness, though less common to list deeply here
            "decorators": decorator_list_str,
            "docstring": ast.get_docstring(node)
        }

        # Visit children to find methods and nested classes directly under this class
        for child_node in node.body:
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_start = child_node.lineno
                method_end = child_node.end_lineno if hasattr(child_node, 'end_lineno') and child_node.end_lineno is not None else method_start
                method_decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else d.id for d in child_node.decorator_list if hasattr(d, 'id') or hasattr(ast, 'unparse')]
                class_info["methods"].append({
                    "type": "async_function" if isinstance(child_node, ast.AsyncFunctionDef) else "function",
                    "name": child_node.name,
                    "start_line": method_start,
                    "end_line": method_end,
                    "args": [arg.arg for arg in child_node.args.args],
                    "decorators": method_decorators,
                    "docstring": ast.get_docstring(child_node)
                })
            # Not recursing into methods here to avoid adding their nested items to the class's method list.
            # The main visitor will catch them if self.generic_visit(node) is called,
            # but we want a structured representation.
            # If we want full recursion for nested items within methods, the structure would be more complex.
            # For now, just listing direct methods of a class.

        self.structures.append(class_info)
        # We call generic_visit to allow finding nested classes, but be mindful of how methods are added.
        # The current method collection is specific to direct children.
        # If generic_visit is called, it will also call visit_FunctionDef for methods,
        # potentially adding them to the top-level self.structures list again.
        # This needs careful handling if we want a strict hierarchy.
        # For now, let's refine by not calling generic_visit on ClassDef if we manually process its body for methods.
        # Instead, explicitly visit other types of children if needed.
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)): # Avoid re-visiting methods here
                 self.visit(item)


def parse_code_structure(code_content: str) -> Dict[str, Any]:
    """
    Parses Python code and extracts a list of top-level structures like functions and classes.

    Args:
        code_content: A string containing Python code.

    Returns:
        A dictionary with a "structures" key containing a list of found structures,
        or an "error" key if parsing fails.
    """
    try:
        tree = ast.parse(code_content)
        visitor = CodeStructureVisitor()
        visitor.visit(tree)
        logger.info(f"Parsed code structure, found {len(visitor.structures)} top-level structures.")
        return {"structures": visitor.structures}
    except SyntaxError as e:
        logger.error(f"Syntax error parsing code: {e}", exc_info=True)
        return {"error": "Syntax error in code.", "details": {"message": str(e.msg), "lineno": e.lineno, "offset": e.offset}}
    except Exception as e:
        logger.error(f"Unexpected error parsing code structure: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during code parsing: {e}"}


def find_contextual_structure(
    code_content: str, line_number: int, column_number: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parses Python code and finds the function or class definition that
    encloses the given line number (and optionally column number).

    Args:
        code_content: A string containing Python code.
        line_number: The 1-based line number of interest.
        column_number: Optional 0-based column number of interest (currently not used for finer grain).

    Returns:
        A dictionary describing the found structure (name, type, start/end lines),
        or {"error": "context not found"} or parsing error details.
    """
    parsed_data = parse_code_structure(code_content)
    if "error" in parsed_data:
        return parsed_data # Propagate parsing error

    # We need a more sophisticated visitor or traversal for finding the *most specific* enclosing scope.
    # The CodeStructureVisitor above gives a flat list.
    # Let's implement a targeted search.

    best_match: Optional[Dict[str, Any]] = None

    try:
        tree = ast.parse(code_content)
    except SyntaxError as e:
        return {"error": "Syntax error in code.", "details": {"message": str(e.msg), "lineno": e.lineno, "offset": e.offset}}
    except Exception as e:
        return {"error": f"Unexpected error during code parsing: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            # end_lineno might not exist for one-liners or be None in some Python versions/constructs
            end_line = getattr(node, 'end_lineno', start_line)
            if end_line is None: end_line = start_line # Fallback

            if start_line <= line_number <= end_line:
                node_type = "unknown"
                if isinstance(node, ast.FunctionDef): node_type = "function"
                elif isinstance(node, ast.AsyncFunctionDef): node_type = "async_function"
                elif isinstance(node, ast.ClassDef): node_type = "class"

                current_match_info = {
                    "type": node_type,
                    "name": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "docstring": ast.get_docstring(node),
                }
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    current_match_info["args"] = [arg.arg for arg in node.args.args]
                    current_match_info["decorators"] = [ast.unparse(d) if hasattr(ast, 'unparse') else d.id for d in node.decorator_list if hasattr(d, 'id') or hasattr(ast, 'unparse')]


                # If this is the first match, or if this match is more specific (smaller range)
                if best_match is None or \
                   (current_match_info["start_line"] >= best_match["start_line"] and \
                    current_match_info["end_line"] <= best_match["end_line"]):
                    best_match = current_match_info

    if best_match:
        logger.info(f"Found contextual structure for line {line_number}: {best_match['type']} {best_match['name']}")
        return {"context": best_match}
    else:
        logger.info(f"No specific function or class context found for line {line_number}.")
        return {"status": "success", "message": "No specific function or class context found for the given line.", "context": None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    sample_code = """
import os

class MySampleClass:
    '''This is a sample class docstring.'''
    class_var = 123

    def __init__(self, name):
        '''Constructor docstring.'''
        self.name = name

    @property
    def decorated_method(self, param1, param2="default"):
        '''A sample method docstring.'''
        # This is a comment inside the method
        local_var = self.name + " " + str(param1)
        if True: # A nested block
            print(local_var)
        return f"Hello, {local_var} from MySampleClass"

    async def async_method(self):
        return "async result"

def top_level_function(x, y):
    '''Top level function docstring'''
    z = x + y
    # Another comment
    return z

# A comment outside any structure
    """

    logger.info("--- Testing parse_code_structure ---")
    structure_result = parse_code_structure(sample_code)
    if "error" in structure_result:
        print(f"Error parsing structure: {structure_result['error']}")
        if "details" in structure_result: print(f"Details: {structure_result['details']}")
    else:
        print("Parsed Structures:")
        for s in structure_result.get("structures", []):
            print(f"  - Type: {s['type']}, Name: {s['name']}, Lines: {s['start_line']}-{s['end_line']}")
            if "args" in s: print(f"    Args: {s['args']}")
            if "decorators" in s and s['decorators']: print(f"    Decorators: {s['decorators']}")
            if "docstring" in s and s['docstring']: print(f"    Docstring: '{s['docstring'][:30]}...'")
            if s['type'] == 'class' and "methods" in s:
                for m in s['methods']:
                    print(f"    - Method: {m['name']}, Lines: {m['start_line']}-{m['end_line']}, Args: {m['args']}")
                    if "decorators" in m and m['decorators']: print(f"      Decorators: {m['decorators']}")


    logger.info("\n--- Testing find_contextual_structure ---")
    test_cases = [
        (8, "Constructor __init__ of MySampleClass"), # Inside __init__
        (13, "Method decorated_method of MySampleClass"), # Inside decorated_method
        (19, "Async method async_method of MySampleClass"), # Inside async_method
        (3, "Class MySampleClass"), # Class definition line
        (22, "Function top_level_function"), # Inside top_level_function
        (27, "Outside any specific function/class (should be None or no specific context)"), # Comment after function
        (1, "Outside (import os)"),
    ]

    for line, desc in test_cases:
        print(f"\nTesting line {line} ({desc}):")
        context_result = find_contextual_structure(sample_code, line)
        if "error" in context_result:
            print(f"  Error: {context_result['error']}")
        elif context_result.get("context"):
            ctx = context_result["context"]
            print(f"  Found: Type: {ctx['type']}, Name: {ctx['name']}, Lines: {ctx['start_line']}-{ctx['end_line']}")
            if "docstring" in ctx and ctx['docstring']: print(f"    Docstring: '{ctx['docstring'][:30]}...'")
        else:
            print(f"  {context_result.get('message', 'No context returned.')}")

    logger.info("\n--- Testing with Syntax Error ---")
    syntax_error_code = """
def my_func(a, b)
    return a + b
    """
    error_structure_result = parse_code_structure(syntax_error_code)
    print(f"Parse structure (syntax error): {error_structure_result}")
    assert "error" in error_structure_result and "Syntax error" in error_structure_result["error"], "Syntax error not caught by parse_code_structure"

    error_context_result = find_contextual_structure(syntax_error_code, 2)
    print(f"Find context (syntax error): {error_context_result}")
    assert "error" in error_context_result and "Syntax error" in error_context_result["error"], "Syntax error not caught by find_contextual_structure"


    logger.info("Code Parser module example finished.")
