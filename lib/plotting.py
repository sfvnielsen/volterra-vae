import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def plot_eyediagram(rx_out: npt.ArrayLike, ax: plt.Axes, Ts: float, sps: int, histogram: bool = False,
                    decimation=10, n_symbol_periods=4, shift=0):
    t = np.arange(shift * Ts, shift * Ts + n_symbol_periods * Ts * sps, Ts)
    discard_n_symbol_periods = 10
    if histogram:
        ax.hist2d(np.tile(t, len(rx_out) // len(t)), rx_out, bins=(n_symbol_periods * sps, 50), cmap='inferno')
    else:
        ax.plot(t, np.reshape(np.roll(rx_out, shift), (-1, sps * n_symbol_periods))[discard_n_symbol_periods:-discard_n_symbol_periods:decimation].T,
                color='crimson', alpha=.1, lw=.5)
        ax.grid(True)
        ax.set_xlim((np.min(t), np.max(t)))