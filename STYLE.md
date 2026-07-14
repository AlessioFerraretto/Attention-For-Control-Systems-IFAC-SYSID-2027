# Code Style and Contributing Guidelines

This repository uses Ruff for Python linting and formatting, Prettier for non-Python configuration and markup files, and autoDocstring for consistent documentation. The goal is to keep the codebase readable, consistent, and easy to maintain.

## Python Coding Standards

Ruff is the primary tool for Python style enforcement. It runs on save and helps keep imports, formatting, and common issues under control.

### Enabled Ruff Rules

The project enforces the following rule sets:

| Rule Code | Description          | Purpose                                                                  |
| :-------- | :------------------- | :----------------------------------------------------------------------- |
| **F**     | Pyflakes             | Catches basic logical errors (e.g., unused variables, undefined names).  |
| **W**     | pycodestyle Warnings | Enforces basic style guidelines and formatting warnings.                 |
| **I**     | isort                | Automatically sorts and groups imports on save.                          |
| **B**     | flake8-bugbear       | Prevents common Python bugs and design issues.                           |
| **UP**    | pyupgrade            | Automatically upgrades syntax to modern Python standards.                |
| **PD**    | pandas-vet           | Enforces best practices for Pandas DataFrames.                           |
| **NPY**   | NumPy-specific rules | Enforces NumPy best practices.                                           |
| **SIM**   | flake8-simplify      | Suggests code simplifications for cleaner logic.                         |
| **E**     | pycodestyle Errors   | Enforces strict PEP 8 style rules to catch formatting errors.            |
| **N**     | pep8-naming          | Enforces PEP 8 naming conventions for variables, functions, and classes. |

### Naming Conventions

- **General Python PEP 8:** Use `snake_case` for variables and functions, and `PascalCase` for classes.
- **Scientific / Paper Exception:** Mathematical variables and architecture components derived directly from research papers may retain their original notation to preserve context (e.g., using `Wd`, or `S6SSM`).

## Documentation and Docstrings

- **Format:** All new classes, methods, and non-trivial logic should be documented using the Google docstring style.
- **Automation:** The `autoDocstring` extension is configured to automatically generate a Google-style template when you press Enter after typing `"""` beneath a function or class definition.
- **Completeness:** You must include an extended summary, document all arguments (Args), and define return types (Returns) for your functions. (You can also use the `/doc` AI skill to help generate these).

## Architectural and Structural Guidelines

When contributing new features, follow the existing architecture and boundaries of the project:

- **Configuration:** Do not hardcode hyperparameters or paths. All experimental setups, dataset defaults, and model variants must be defined in the Hydra YAML files located in `configs/`.
- **Modularity:** Use the existing factory patterns (`datasets.factory`, `models.factory`). If you add a new SSM variant or dataset, register it within the appropriate factory rather than writing bespoke initialization logic.
- **PyTorch Modules:** Keep `nn.Module` components isolated.
- **Device Agnostic:** Ensure all tensor operations utilize the device resolved by `utils.environment.get_torch_device()` to maintain compatibility across CUDA, MPS, and CPU.

## Pre-Commit Checklist

Before committing changes or opening a pull request, make sure you have:

1. Saved your files to trigger Ruff and Prettier formatting.
2. Resolved any remaining Ruff warnings, especially unused imports or naming issues that are not covered by the paper-based exception.
3. Added Google-style docstrings for new classes and methods.
4. Added or updated experiment configurations in [configs/experiment/](configs/experiment/) when needed.
5. Verified that the change still works with a basic run, such as a small training or smoke test.
