"""Compatibility entry point for ``python -m endzone.train.train``.

The canonical module is ``endzone.modeling.train``. This shim keeps the
documented command and the same flags.
"""

from endzone.modeling.train import main

if __name__ == "__main__":
    main()
