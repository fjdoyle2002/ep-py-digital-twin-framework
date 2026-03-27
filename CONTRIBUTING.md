# Contributing to ep-py-digital-twin-framework

Thank you for your interest in contributing. This document describes the process for reporting issues and submitting changes.

## Reporting Issues

Use the GitHub Issues tracker to report bugs or request features. When reporting a bug, please include:

- Your operating system and Python version
- Your EnergyPlus version
- A minimal description of what you expected vs. what happened
- Relevant log output (with any credentials removed)

## Submitting Changes

1. Fork the repository on GitHub.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes. Keep commits focused and write clear commit messages.
4. Ensure your changes do not introduce credentials, proprietary building models, or compiled Python files (`.pyc`, `__pycache__`).
5. Open a pull request against `main` with a description of what your change does and why.

## Code Style

- Follow PEP 8 conventions for Python code.
- Keep module responsibilities aligned with the five-layer architecture described in the README and the associated paper.
- Custom building-specific logic belongs in the `custom/` layer, not in core modules.

## Configuration and Credentials

Never commit credentials, passwords, or API keys. Use `config.ini.example` as the template for any configuration examples, with placeholder values for all sensitive fields.

The `.gitignore` excludes `config.ini` by default precisely for this reason.

## License

By contributing, you agree that your contributions will be licensed under the MIT License that covers this project.
