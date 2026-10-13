EPS = 1e-8


def setup_globals(cfg, log):
    global EPS
    if hasattr(cfg, "model.eps"):
        EPS = cfg.model.eps
    else:
        log.warning(f"eps not found, using default: {EPS}")
