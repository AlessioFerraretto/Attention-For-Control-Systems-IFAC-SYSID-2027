EPS = 1e-8


def setup_globals(cfg, log):
    global EPS
    EPS = cfg.get("model.eps", 1e-8)
