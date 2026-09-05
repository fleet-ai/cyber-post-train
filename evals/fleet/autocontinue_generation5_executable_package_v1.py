"""Generation-5 executable package interface.

Generation 5 has one authority-bound package, rather than a second semantic-only
package.  This module preserves the stable executable-package import surface
while delegating to that single package implementation.
"""

from evals.fleet.autocontinue_generation5_authority_package_v1 import (
    CORE_A_NAME,
    CORE_A_PATHS,
    CORE_B_NAME,
    CORE_B_PATHS,
    KUBERNETES_OBJECT_LIMIT,
    MODEL_NAMES,
    MODEL_PATHS,
    PACKAGE_OBJECT_LIMIT,
    SCHEMA,
    TRANSITIVE_SPEC_PATHS,
    build_package,
    canonical,
    data_key,
    file_sha256,
    sha256,
    verify_mounted,
)

__all__ = [
    "CORE_A_NAME",
    "CORE_A_PATHS",
    "CORE_B_NAME",
    "CORE_B_PATHS",
    "KUBERNETES_OBJECT_LIMIT",
    "MODEL_NAMES",
    "MODEL_PATHS",
    "PACKAGE_OBJECT_LIMIT",
    "SCHEMA",
    "TRANSITIVE_SPEC_PATHS",
    "build_package",
    "canonical",
    "data_key",
    "file_sha256",
    "sha256",
    "verify_mounted",
]
