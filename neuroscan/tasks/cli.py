"""Shared CLI runner boilerplate for neuroscan task entrypoints."""
import logging


class Cli:
    """Common setup shared by every task runner's ``main()``."""

    @staticmethod
    def setup_logging() -> None:
        """Configure root logging and quiet the noisy EEG/fNIRS libraries."""
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        for lib_name in ("mne", "moabb", "braindecode"):
            logging.getLogger(lib_name).setLevel(logging.WARNING)
