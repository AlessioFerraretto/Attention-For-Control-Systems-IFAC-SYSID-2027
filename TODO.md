Parametri del Kernel:

- alpha = Uniform(alpha_min, alpha_max)
- c = Uniform(c_min, c_max)
- eta = alpha + (c^max(i,j))
- SNR = white noise (per ora usiamo una Gaussiana)

Tasks:

- impulse response (problema malposto -> va regolarizzato)
- future prediction (partiamo da questo) dati (u_past, y_past) dato (u_future) predire (y_future)
- hyper parameter
