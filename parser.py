import re
import numpy as np


def parse_input_file(filepath):
    """Parse a MATLAB-style input file into a dict of parameters."""
    params = {}
    ns = {"np": np}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' not in line or line.startswith('%'):
                continue
            if '%' in line:
                line = line[:line.index('%')]
            varname = line[:line.index('=')].strip()
            valstr = line[line.index('=') + 1:].strip().rstrip(';')
            val = _eval_matlab_value(valstr, ns)
            params[varname] = val
            ns[varname] = val
    return params


def _eval_matlab_value(s, ns=None):
    """Evaluate a simple MATLAB expression to a Python value."""
    s = s.strip()
    if s.startswith("'") and s.endswith("'"):
        return s.strip("'")
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        items = [x.strip().strip("'") for x in inner.split(',')]
        return items
    if ns is None:
        ns = {"np": np}
    s_py = s.replace('linspace', 'np.linspace')
    s_py = re.sub(r'\[([^\]]+)\]', _matlab_array_to_python, s_py)
    try:
        return eval(s_py, {"__builtins__": {}}, ns)
    except Exception:
        return s


def _matlab_array_to_python(match):
    """Convert MATLAB array `[1 2 3]` -> `np.array([1, 2, 3])`."""
    inner = match.group(1).strip()
    parts = inner.split()
    if len(parts) > 1 and all(_is_number(p) for p in parts):
        return 'np.array([' + ', '.join(parts) + '])'
    return match.group(0)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
