"""CareerOS AI source package.

This is the root Python package for the CareerOS AI application. All
application logic lives under this namespace so it can be imported as::

    from src.config.settings import MODULES
    from src.core.state import init_state

Subpackages
-----------
- ``src.config``      : app metadata & module registry.
- ``src.core``        : session state, shared utilities.
- ``src.ui``          : layout, reusable components, page builders.
- ``src.services``    : cross-cutting services (LLM client, etc.).
- ``src.modules``     : one package per planned feature.
"""

__version__ = "0.1.0"
