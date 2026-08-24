"""DELTA's tested SINQ implementation.

This package is vendored from the copy installed in the original ``delta``
environment. It intentionally does not resolve or import an external ``sinq``
package.
"""

from .quantizer import Quantizer

__all__ = ["Quantizer"]
