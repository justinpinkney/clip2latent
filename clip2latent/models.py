

def load_sg(network_pkl):
    import sys
    sys.path.append("stylegan3")
    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    return G