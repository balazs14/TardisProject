import pandas as pd
from io import StringIO
import ast
import inspect
import os
import re
import warnings


def df_from_string(string, sep=r'\s+', **kwargs):
    return pd.read_csv(StringIO(string), sep=sep, **kwargs)


def _df_to_string(df):
    if isinstance(df, str):
        return df if df.endswith('\n') else df + '\n'
    return df.to_string() + '\n'


def _normalize_snapshot(text):
    return text.replace(' ', '').replace('\n', '')


def _find_caller_function(tree, function_name, call_lineno):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if node.lineno <= call_lineno <= node.end_lineno:
                return node
    return None


def _find_assignment_before_line(function_node, variable_name, call_lineno):
    chosen = None
    for stmt in function_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if stmt.lineno > call_lineno:
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in stmt.targets):
            continue
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            chosen = stmt
    return chosen


def _rewrite_inline_snapshot(source_path, function_name, call_lineno, variable_name, new_value):
    with open(source_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)
    function_node = _find_caller_function(tree, function_name, call_lineno)
    if function_node is None:
        return False

    assign_node = _find_assignment_before_line(function_node, variable_name, call_lineno)
    if assign_node is None:
        return False

    lines = source.splitlines(keepends=True)
    indent = re.match(r'\s*', lines[assign_node.lineno - 1]).group(0)
    escaped = new_value.replace('"""', '\\"""')
    replacement = f'{indent}{variable_name} = """{escaped}"""\n'

    lines[assign_node.lineno - 1:assign_node.end_lineno] = [replacement]

    with open(source_path, 'w') as f:
        f.writelines(lines)
    return True


def _line_col_to_offset(lines, line_no, col_no):
    return sum(len(line) for line in lines[: line_no - 1]) + col_no


def _find_assert_df_equal_call(tree, function_name, call_lineno):
    function_node = _find_caller_function(tree, function_name, call_lineno)
    if function_node is None:
        return None

    chosen = None
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue

        fn = node.func
        is_assert_df_equal = (
            (isinstance(fn, ast.Name) and fn.id == 'assert_df_equal')
            or (isinstance(fn, ast.Attribute) and fn.attr == 'assert_df_equal')
        )
        if not is_assert_df_equal:
            continue

        if not (node.lineno <= call_lineno <= node.end_lineno):
            continue

        if chosen is None or node.lineno >= chosen.lineno:
            chosen = node

    return chosen


def _rewrite_assert_df_equal_inline_second_arg(source_path, function_name, call_lineno, new_value):
    with open(source_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)
    call_node = _find_assert_df_equal_call(tree, function_name, call_lineno)
    if call_node is None or len(call_node.args) < 2:
        return False

    second_arg = call_node.args[1]
    if not (isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str)):
        return False

    lines = source.splitlines(keepends=True)
    start = _line_col_to_offset(lines, second_arg.lineno, second_arg.col_offset)
    end = _line_col_to_offset(lines, second_arg.end_lineno, second_arg.end_col_offset)

    escaped = new_value.replace('"""', '\\"""')
    replacement = f'"""{escaped}"""'

    updated = source[:start] + replacement + source[end:]
    with open(source_path, 'w') as f:
        f.write(updated)
    return True

def assert_df_equal(df, df_good, expected_var_name=None, update=None):
    str_df = _df_to_string(df)

    if isinstance(df_good, pd.DataFrame):
        str_good = _df_to_string(df_good)
    else:
        str_good = _df_to_string(df_good)

    comp_df = _normalize_snapshot(str_df)
    comp_good = _normalize_snapshot(str_good)

    if comp_df == comp_good:
        return False

    if expected_var_name is not None:
        if update is None:
            update = os.getenv('UPDATE_DF_SNAPSHOTS', '').lower() in {'1', 'true', 'yes', 'on'}

        if update:
            frame = inspect.currentframe().f_back
            source_path = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            call_lineno = frame.f_lineno
            updated = _rewrite_inline_snapshot(source_path, function_name, call_lineno, expected_var_name, str_df)
            if not updated:
                raise AssertionError(
                    f"Could not auto-update inline snapshot variable '{expected_var_name}' in {source_path}."
                )
            warnings.warn(
                f"Updated inline snapshot '{expected_var_name}' in {source_path}. "
                "This test would have failed without update mode.",
                stacklevel=2,
            )
            return True
    else:
        if update is None:
            update = os.getenv('UPDATE_DF_SNAPSHOTS', '').lower() in {'1', 'true', 'yes', 'on'}

        if update:
            frame = inspect.currentframe().f_back
            source_path = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            call_lineno = frame.f_lineno
            updated = _rewrite_assert_df_equal_inline_second_arg(source_path, function_name, call_lineno, str_df)
            if not updated:
                raise AssertionError(
                    "Could not auto-update inline second argument for assert_df_equal. "
                    "Ensure the second argument is a string literal or pass expected_var_name."
                )
            warnings.warn(
                f"Updated inline assert_df_equal snapshot in {source_path}. "
                "This test would have failed without update mode.",
                stacklevel=2,
            )
            return True

    with open('/tmp/tmp_df', 'w') as f:
        f.write(str_df)

    message = '\n\nRESULT SHOULD HAVE BEEN: \n'
    message += str_good
    message += '\nBUT IT WAS: \n'
    message += str_df
    if expected_var_name is not None:
        message += "\n\nRe-run with UPDATE_DF_SNAPSHOTS=1 to auto-update inline snapshots in test code."

    assert comp_df == comp_good, message

