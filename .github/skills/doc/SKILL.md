---
name: doc
user-invocable: true
description: "Document Python code using Google Python Style docstrings using the requested style."
---

# Google Python Docstring Documentation

This skill helps document Python code using the Google Python Style guide and enforces a specific docstring layout.

## What it does

From most prioritized to least:

- Preserves existing code structure while enhancing documentation clarity.
- Adds or improves docstrings for Python modules, classes, methods, and functions.
- Use the exact section headings with trailing colons and the same underline style shown in the template.
- Follow the Ruff-style formatting rules.
- Follow the Google-style formatting rules.
- Follows best practices for writing clear and concise docstrings that are helpful for code maintainers and users.

## Quality checklist

- [ ] Docstrings use the requested formatting and indentation.
- [ ] Existing code semantics and behavior are preserved.

## Exact template

When documenting a file, follow this docstring layout where words in parentheses specify the sections where applicable. Use the correct spacing and indentation depending on code.

```python
"""Title

Description text describing the module/class/function purpose.

Parameters: (only for functions/methods and classes with an __init__ method)
  name : type
    Description of the parameter.

Attributes: (only for classes)
  name : type
    Description of the attribute.

Returns: (only for functions/methods)
  type: Description of the return value.

Raises: (only for functions/methods that raise exceptions)
  ExceptionType: Description of the error condition.

Examples: (only if explicitly requested by the user)
  example in code

Notes: (only if explicitly requested by the user)
  Additional notes or clarifications.

"""
```
