import logging, sys, pathlib

class _StreamToLogger:
    """File-like object that forwards writes to a logger."""
    def __init__(self, logger: logging.Logger, level: int):
        self._logger, self._level = logger, level
    def write(self, message: str):
        msg = message.rstrip()
        if msg:
            self._logger.log(self._level, msg)
    def flush(self):   # 필요 인터페이스
        pass

def init_snakemake_logger(snakemake, *,
                          name: str = "vaft",
                          std_opt: bool = True) -> logging.Logger:
    """
    Create (or reuse) a logger that writes to both console and the rule's
    log file.  
    Parameters
    ----------
    snakemake : object
        Snakemake-injected object available inside `script:` files.
    name : str
        Logger name (default "vaft").
    std_opt : bool
        If True  → redirect sys.stdout → logger at INFO level,
                   redirect sys.stderr → logger at ERROR level.
        If False → leave stdout/stderr unchanged.
    """
    # ---- 1) build / reuse logger -----------------------------------------
    log_path = pathlib.Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:         # idempotent: add handlers only once
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        sh = logging.StreamHandler(sys.stdout)  # console
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_path)      # rule log file
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # ---- 2) optional stdout/stderr redirect ------------------------------
    if std_opt:
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

    return logger