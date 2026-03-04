"""
plot_helper.py
==============
Helper class for easy plotting of EmbedSim simulation results.

This module provides a clean interface for visualizing simulation data
without modifying the original EmbedSim class. It includes methods for
quick plotting, component analysis, signal comparison, and XY plots.

Author: EmbedSim Framework
Version: 1.0.0
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Union, Any, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .simulation_engine import EmbedSim


class PlotHelper:
    """
    Helper class for easy plotting of EmbedSim simulation results.

    Takes a simulation object and provides convenient plotting methods
    without modifying the original EmbedSim class.

    Attributes
    ----------
    sim : EmbedSim
        The simulation object containing scope data
    t : np.ndarray
        Time vector from the simulation

    Examples
    --------
    >>> # This is just a class definition - see create_plotter() for usage
    >>> pass
    """

    def __init__(self, sim: EmbedSim) -> None:
        """
        Initialize the plot helper with a simulation object.

        Parameters
        ----------
        sim : EmbedSim
            The simulation object containing scope data. Must have a scope
            attribute with t and data properties.

        Raises
        ------
        AttributeError
            If the simulation object doesn't have a scope attribute
        """
        self.sim = sim
        self.t: np.ndarray = np.array(sim.scope.t) if hasattr(sim.scope, 't') else np.array([])

    def _check_data(self) -> bool:
        """
        Check if there's data to plot.

        Returns
        -------
        bool
            True if data exists, False otherwise
        """
        if len(self.t) == 0:
            print("⚠ No data recorded. Make sure simulation ran and scope.add() was used.")
            return False
        return True

    def _get_time_mask(self, time_range: Optional[Tuple[float, float]] = None) -> Union[slice, np.ndarray, None]:
        """
        Get mask for time range selection.

        Parameters
        ----------
        time_range : tuple of float, optional
            Time range as (t_start, t_end)

        Returns
        -------
        slice or np.ndarray or None
            Mask for indexing time and data arrays, or None if range invalid
        """
        if time_range is None:
            return slice(None)
        t_start, t_end = time_range
        mask = (self.t >= t_start) & (self.t <= t_end)
        if not np.any(mask):
            print(f"⚠ No data in time range [{t_start}, {t_end}]")
            return None
        return mask

    def info(self) -> None:
        """
        Print information about all recorded signals.

        Shows:
        - List of all signal names
        - Signal dimensions (number of components)
        - Time span and number of samples
        - Basic statistics (min, max, mean) for each signal

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> plotter.info()  # doctest: +SKIP
        """
        if not self._check_data():
            return

        print("\n" + "=" * 70)
        print("📊 RECORDED SIGNAL INFORMATION")
        print("=" * 70)

        print(f"\n📈 Time data:")
        print(f"   Samples: {len(self.t)}")
        print(f"   Start:   {self.t[0]:.4f} s")
        print(f"   End:     {self.t[-1]:.4f} s")
        print(f"   Step:    {self.t[1] - self.t[0]:.6f} s")

        # Group signals by block
        blocks: Dict[str, List[str]] = {}
        for key in self.sim.scope.data.keys():
            if '[' in key:
                block = key.split('[')[0]
                if block not in blocks:
                    blocks[block] = []
                blocks[block].append(key)

        print(f"\n📊 Blocks and signals:")
        for block, signals in sorted(blocks.items()):
            n_comp = len(signals)
            comp_indices = [s.rsplit('[', 1)[1].rstrip(']') for s in signals]
            comp_str = ', '.join(comp_indices)
            print(f"\n   📦 {block}: {n_comp} component(s) [{comp_str}]")

            # Print statistics for each component
            for sig in sorted(signals):
                data = np.array(self.sim.scope.data[sig])
                print(f"      └─ {sig}: min={data.min():.3f}, "
                      f"max={data.max():.3f}, mean={data.mean():.3f}")

        print("\n" + "=" * 70)

    def list_signals(self) -> None:
        """
        Print a simple list of all available signals.

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> plotter.list_signals()  # doctest: +SKIP
        """
        if not self._check_data():
            return

        print("\n📋 Available signals:")
        for i, key in enumerate(sorted(self.sim.scope.data.keys())):
            print(f"   {i+1:2d}. {key}")
        print()

    def _resolve_signals(self, signals: Optional[Union[str, List[str]]]) -> List[str]:
        """
        Resolve signal specifications to actual signal names.

        Parameters
        ----------
        signals : str or list of str or None
            Signal specification(s)

        Returns
        -------
        list of str
            List of actual signal names found in the data
        """
        if signals is None:
            return list(self.sim.scope.data.keys())

        if isinstance(signals, str):
            # Direct component key: last segment is a numeric index e.g. "...[0]"
            # Detect by checking if last bracketed part is a digit
            last_bracket = signals.rsplit('[', 1)[-1].rstrip(']')
            is_component_key = last_bracket.isdigit() and signals in self.sim.scope.data

            if is_component_key:
                return [signals]
            elif signals in self.sim.scope.data:
                # Exact match (non-numeric suffix) — return as-is
                return [signals]
            else:
                # Treat as label prefix — find all component keys that start with it
                pattern = f"{signals}["
                found = [key for key in self.sim.scope.data.keys()
                        if key.startswith(pattern)]
                if found:
                    return sorted(found)
                else:
                    print(f"⚠ No signals found for '{signals}'")
                    return []

        if isinstance(signals, list):
            result = []
            for sig in signals:
                if sig in self.sim.scope.data:
                    result.append(sig)
                else:
                    print(f"⚠ Warning: Signal '{sig}' not found, skipping")
            return result

        return []

    def easyplot(self,
                 signals: Optional[Union[str, List[str]]] = None,
                 title: Optional[str] = None,
                 time_range: Optional[Tuple[float, float]] = None,
                 figsize: Tuple[int, int] = (12, 6),
                 style: str = 'default',
                 save_path: Optional[str] = None,
                 legend: bool = True,
                 grid: bool = True,
                 linewidth: float = 1.5,
                 alpha: float = 0.8,
                 ax: Optional[Axes] = None) -> Tuple[Optional[Figure], Optional[Axes]]:
        """
        Easy plotting function for quick visualization of recorded signals.

        Parameters
        ----------
        signals : str or list of str or None, optional
            - If str: Name of a block or signal pattern (e.g., "gain" or "gain[0]")
            - If list: List of signal names or patterns
            - If None: Plot all recorded signals (default)

        title : str, optional
            Plot title. If None, automatically generated from signals.

        time_range : tuple of float, optional
            Time range to plot as (t_start, t_end). If None, plot all time.

        figsize : tuple of int, optional
            Figure size as (width, height) in inches. Default (12, 6).

        style : str, optional
            Matplotlib style to use. Default 'default'.
            Options: 'default', 'ggplot', 'seaborn', 'dark_background', etc.

        save_path : str, optional
            If provided, save the figure to this path (e.g., 'plot.png').

        legend : bool, optional
            Whether to show legend. Default True.

        grid : bool, optional
            Whether to show grid. Default True.

        linewidth : float, optional
            Line width for plots. Default 1.5.

        alpha : float, optional
            Transparency of lines. Default 0.8.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if successful, None otherwise
        ax : matplotlib.axes.Axes or None
            The axes object if successful, None otherwise

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> # Plot all signals
        >>> plotter.easyplot()  # doctest: +SKIP
        >>>
        >>> # Plot specific block (all its components)
        >>> plotter.easyplot('clarke')  # doctest: +SKIP
        >>>
        >>> # Plot specific component
        >>> plotter.easyplot('clarke[0]')  # doctest: +SKIP
        """
        if not self._check_data():
            return None, None

        # Apply style
        with plt.style.context(style):
            # Use injected ax if provided (for subplots), else create own figure
            if ax is not None:
                fig = ax.get_figure()
            else:
                fig, ax = plt.subplots(figsize=figsize)

            # Apply time range
            mask = self._get_time_mask(time_range)
            if mask is None:
                return None, None

            # Handle time data based on mask type
            if isinstance(mask, slice):
                t_plot = self.t
            else:
                t_plot = self.t[mask]

            # Collect signals to plot
            signals_to_plot = self._resolve_signals(signals)

            if not signals_to_plot:
                print("⚠ No signals to plot")
                return None, None

            # Plot each signal
            colors = plt.cm.tab10(np.linspace(0, 1, len(signals_to_plot)))
            for i, sig_name in enumerate(signals_to_plot):
                data = np.array(self.sim.scope.data[sig_name])
                if not isinstance(mask, slice):
                    data = data[mask]

                # Create nice label
                if '[' in sig_name:
                    block = sig_name.rsplit('[', 1)[0]
                    comp  = sig_name.rsplit('[', 1)[1].rstrip(']')
                    label = f"{block} [{comp}]"
                else:
                    label = sig_name

                ax.plot(t_plot, data, label=label, linewidth=linewidth,
                       color=colors[i % len(colors)], alpha=alpha)

            # Set labels and title
            ax.set_xlabel('Time [s]', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)

            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                if len(signals_to_plot) == 1:
                    base = signals_to_plot[0].split('[')[0]
                    ax.set_title(f'Signal: {base}', fontsize=14, fontweight='bold')
                elif len(signals_to_plot) <= 3:
                    ax.set_title('Multiple Signals', fontsize=14, fontweight='bold')

            # Add grid and legend
            if grid:
                ax.grid(True, alpha=0.3, linestyle='--')
            if legend and len(signals_to_plot) <= 10:
                ax.legend(loc='best', fontsize=10)
            elif len(signals_to_plot) > 10:
                print(f"ℹ Too many signals ({len(signals_to_plot)}) to show legend")

            # Only manage figure lifecycle when we created the figure.
            # If ax was injected externally the caller handles show/save/layout.
            _owns_figure = (ax is None)
            if _owns_figure:
                plt.tight_layout()
                if save_path:
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"✅ Plot saved to: {save_path}")
                plt.show()

            return fig, ax

    def plot_components(self,
                        block_name: str,
                        time_range: Optional[Tuple[float, float]] = None,
                        figsize: Tuple[int, int] = (12, 8),
                        same_scale: bool = False,
                        colors: Optional[List[str]] = None,
                        grid: bool = True) -> Tuple[Optional[Figure], Optional[np.ndarray]]:
        """
        Plot all components of a vector block in separate subplots.

        Parameters
        ----------
        block_name : str
            Name of the block to plot (all its components)
        time_range : tuple of float, optional
            Time range as (t_start, t_end)
        figsize : tuple of int, optional
            Figure size (width, height). Default (12, 8)
        same_scale : bool, optional
            If True, all subplots share the same y-axis limits. Default False
        colors : list of str, optional
            List of colors for each component. If None, uses default colormap
        grid : bool, optional
            Whether to show grid. Default True

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if successful, None otherwise
        axes : numpy.ndarray of matplotlib.axes.Axes or None
            Array of axes objects if successful, None otherwise

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> # Plot all three phases of a generator
        >>> plotter.plot_components('3phase_gen')  # doctest: +SKIP
        >>>
        >>> # Plot with same scale for comparison
        >>> plotter.plot_components('clarke', same_scale=True)  # doctest: +SKIP
        """
        if not self._check_data():
            return None, None

        # Find all components for this block
        pattern = f"{block_name}["
        components = sorted([key for key in self.sim.scope.data.keys()
                            if key.startswith(pattern)])

        if not components:
            print(f"⚠ No components found for block '{block_name}'")
            return None, None

        n_components = len(components)

        # Determine grid layout
        if n_components <= 3:
            n_rows, n_cols = n_components, 1
        else:
            n_cols = 2
            n_rows = (n_components + 1) // 2

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        # Apply time range
        mask = self._get_time_mask(time_range)
        if mask is None:
            return None, None

        # Handle time data based on mask type
        if isinstance(mask, slice):
            t_plot = self.t
        else:
            t_plot = self.t[mask]

        # Find global y limits if same_scale
        if same_scale:
            y_min, y_max = float('inf'), float('-inf')
            for comp_name in components:
                data = np.array(self.sim.scope.data[comp_name])
                if not isinstance(mask, slice):
                    data = data[mask]
                y_min = min(y_min, data.min())
                y_max = max(y_max, data.max())
            # Add some padding
            padding = (y_max - y_min) * 0.1
            y_limits = (y_min - padding, y_max + padding)

        # Setup colors
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_components))

        # Plot each component
        for i, comp_name in enumerate(components):
            data = np.array(self.sim.scope.data[comp_name])
            if not isinstance(mask, slice):
                data = data[mask]

            # Extract component index for title
            comp_idx = comp_name.rsplit('[', 1)[1].rstrip(']')

            axes_flat[i].plot(t_plot, data, linewidth=1.5,
                            color=colors[i % len(colors)] if isinstance(colors, list) else colors[i])
            axes_flat[i].set_title(f'Component {comp_idx}', fontsize=11)
            axes_flat[i].set_ylabel('Amplitude')
            if grid:
                axes_flat[i].grid(True, alpha=0.3)

            if same_scale:
                axes_flat[i].set_ylim(y_limits)

            # Only label x-axis on bottom plots
            if i >= n_components - n_cols:
                axes_flat[i].set_xlabel('Time [s]')

        # Hide unused subplots
        for j in range(len(components), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f'Block: {block_name} (All Components)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig, axes

    def compare(self,
                signal1: str,
                signal2: str,
                time_range: Optional[Tuple[float, float]] = None,
                figsize: Tuple[int, int] = (12, 5),
                titles: Optional[List[str]] = None,
                colors: Optional[List[str]] = None,
                grid: bool = True) -> Tuple[Optional[Figure], Optional[np.ndarray]]:
        """
        Compare two signals in side-by-side plots.

        Parameters
        ----------
        signal1, signal2 : str
            Signal names to compare
        time_range : tuple of float, optional
            Time range to plot
        figsize : tuple of int, optional
            Figure size. Default (12, 5)
        titles : list of str, optional
            Custom titles for the two subplots
        colors : list of str, optional
            Colors for the two signals
        grid : bool, optional
            Whether to show grid. Default True

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if successful, None otherwise
        axes : numpy.ndarray of matplotlib.axes.Axes or None
            Array of axes objects if successful, None otherwise

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> plotter.compare('clarke[0]', 'clarke[1]')  # doctest: +SKIP
        >>> plotter.compare('3phase_gen[0]', 'clarke[0]',
        ...                 titles=['Phase A', 'Alpha'])  # doctest: +SKIP
        """
        if not self._check_data():
            return None, None

        # Check if signals exist
        if signal1 not in self.sim.scope.data:
            print(f"⚠ Signal '{signal1}' not found")
            return None, None
        if signal2 not in self.sim.scope.data:
            print(f"⚠ Signal '{signal2}' not found")
            return None, None

        # Apply time range
        mask = self._get_time_mask(time_range)
        if mask is None:
            return None, None

        # Handle time data based on mask type
        if isinstance(mask, slice):
            t_plot = self.t
        else:
            t_plot = self.t[mask]

        data1 = np.array(self.sim.scope.data[signal1])
        data2 = np.array(self.sim.scope.data[signal2])
        if not isinstance(mask, slice):
            data1 = data1[mask]
            data2 = data2[mask]

        # Setup colors
        if colors is None:
            colors = ['blue', 'red']

        # Setup titles
        if titles is None:
            titles = [signal1, signal2]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes

        # Plot individual signals
        ax1.plot(t_plot, data1, color=colors[0], linewidth=1.5)
        ax1.set_title(titles[0])
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        if grid:
            ax1.grid(True, alpha=0.3)

        ax2.plot(t_plot, data2, color=colors[1], linewidth=1.5)
        ax2.set_title(titles[1])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Amplitude')
        if grid:
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, axes

    def xy_plot(self,
                x_signal: str,
                y_signal: str,
                time_range: Optional[Tuple[float, float]] = None,
                figsize: Tuple[int, int] = (8, 8),
                title: Optional[str] = None,
                color: str = 'blue',
                linewidth: float = 1.0,
                alpha: float = 0.7,
                grid: bool = True,
                equal_axes: bool = True) -> Tuple[Optional[Figure], Optional[Axes]]:
        """
        Create an XY plot (Lissajous figure) from two signals.

        Useful for phase relationship visualization (e.g., alpha vs beta).

        Parameters
        ----------
        x_signal, y_signal : str
            Signal names for x and y axes
        time_range : tuple of float, optional
            Time range to plot
        figsize : tuple of int, optional
            Figure size. Default (8, 8)
        title : str, optional
            Plot title. If None, auto-generated
        color : str, optional
            Line color. Default 'blue'
        linewidth : float, optional
            Line width. Default 1.0
        alpha : float, optional
            Line transparency. Default 0.7
        grid : bool, optional
            Whether to show grid. Default True
        equal_axes : bool, optional
            If True, sets equal aspect ratio. Default True

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if successful, None otherwise
        ax : matplotlib.axes.Axes or None
            The axes object if successful, None otherwise

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> # Alpha vs Beta (should show a circle for balanced 3-phase)
        >>> plotter.xy_plot('clarke[0]', 'clarke[1]')  # doctest: +SKIP
        >>>
        >>> # Phase A vs Phase B (should show 120° phase shift)
        >>> plotter.xy_plot('3phase_gen[0]', '3phase_gen[1]')  # doctest: +SKIP
        """
        if not self._check_data():
            return None, None

        # Check if signals exist
        if x_signal not in self.sim.scope.data:
            print(f"⚠ Signal '{x_signal}' not found")
            return None, None
        if y_signal not in self.sim.scope.data:
            print(f"⚠ Signal '{y_signal}' not found")
            return None, None

        # Apply time range
        mask = self._get_time_mask(time_range)
        if mask is None:
            return None, None

        x_data = np.array(self.sim.scope.data[x_signal])
        y_data = np.array(self.sim.scope.data[y_signal])
        if not isinstance(mask, slice):
            x_data = x_data[mask]
            y_data = y_data[mask]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_data, y_data, color=color, linewidth=linewidth, alpha=alpha)

        # Set labels and title
        ax.set_xlabel(x_signal)
        ax.set_ylabel(y_signal)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'XY Plot: {x_signal} vs {y_signal}')

        # Add grid and axes
        if grid:
            ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)

        if equal_axes:
            ax.axis('equal')

        plt.tight_layout()
        plt.show()

        return fig, ax

    def fft_plot(self,
                 signal_name: str,
                 time_range: Optional[Tuple[float, float]] = None,
                 figsize: Tuple[int, int] = (12, 5),
                 max_freq: Optional[float] = None,
                 window: str = 'hanning',
                 title: Optional[str] = None,
                 log_scale: bool = False) -> Tuple[Optional[Figure], Optional[np.ndarray]]:
        """
        Plot frequency spectrum (FFT) of a signal.

        Parameters
        ----------
        signal_name : str
            Name of the signal to analyze
        time_range : tuple of float, optional
            Time range to use for FFT
        figsize : tuple of int, optional
            Figure size. Default (12, 5)
        max_freq : float, optional
            Maximum frequency to display
        window : str, optional
            Window function to apply ('hanning', 'hamming', 'blackman', or None)
        title : str, optional
            Plot title
        log_scale : bool, optional
            If True, use logarithmic scale for y-axis

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if successful, None otherwise
        axes : numpy.ndarray of matplotlib.axes.Axes or None
            Array of axes objects if successful, None otherwise

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> plotter.fft_plot('clarke[0]', max_freq=200)  # doctest: +SKIP
        """
        if not self._check_data():
            return None, None

        # Check if signal exists
        if signal_name not in self.sim.scope.data:
            print(f"⚠ Signal '{signal_name}' not found")
            return None, None

        # Apply time range
        mask = self._get_time_mask(time_range)
        if mask is None:
            return None, None

        data = np.array(self.sim.scope.data[signal_name])
        if not isinstance(mask, slice):
            data = data[mask]
            t_segment = self.t[mask]
        else:
            t_segment = self.t

        # Calculate FFT
        n = len(data)
        dt = t_segment[1] - t_segment[0]
        fs = 1.0 / dt  # Sampling frequency

        # Apply window if specified
        if window == 'hanning':
            window_func = np.hanning(n)
        elif window == 'hamming':
            window_func = np.hamming(n)
        elif window == 'blackman':
            window_func = np.blackman(n)
        else:
            window_func = np.ones(n)

        windowed_data = data * window_func

        # Compute FFT
        fft_vals = np.fft.fft(windowed_data)
        fft_freqs = np.fft.fftfreq(n, dt)

        # Take positive frequencies only
        pos_mask = fft_freqs >= 0
        freqs = fft_freqs[pos_mask]
        magnitude = np.abs(fft_vals[pos_mask])

        # Apply frequency limit
        if max_freq:
            freq_mask = freqs <= max_freq
            freqs = freqs[freq_mask]
            magnitude = magnitude[freq_mask]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot time domain
        ax1.plot(t_segment, data, 'b-', linewidth=1.0)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Time Domain: {signal_name}')
        ax1.grid(True, alpha=0.3)

        # Plot frequency domain
        if log_scale:
            ax2.semilogy(freqs, magnitude, 'r-', linewidth=1.5)
        else:
            ax2.plot(freqs, magnitude, 'r-', linewidth=1.5)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Magnitude')
        if title:
            ax2.set_title(title)
        else:
            ax2.set_title(f'Frequency Spectrum: {signal_name}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, np.array([ax1, ax2])

    def save_all_plots(self,
                       prefix: str = 'plot',
                       format: str = 'png',
                       dpi: int = 150) -> None:
        """
        Save all available plots to files.

        Parameters
        ----------
        prefix : str, optional
            Prefix for filenames. Default 'plot'
        format : str, optional
            Image format ('png', 'pdf', 'svg', etc.). Default 'png'
        dpi : int, optional
            Resolution for raster formats. Default 150

        Examples
        --------
        >>> # Assume 'plotter' is a PlotHelper instance with data
        >>> plotter.save_all_plots('clarke_test', format='pdf')  # doctest: +SKIP
        """
        if not self._check_data():
            return

        print(f"\n💾 Saving all plots with prefix '{prefix}'...")

        # Get all blocks
        blocks = set()
        for key in self.sim.scope.data.keys():
            if '[' in key:
                blocks.add(key.split('[')[0])

        # Save component plots for each block
        for block in sorted(blocks):
            filename = f"{prefix}_{block}.{format}"
            fig, _ = self.plot_components(block, figsize=(10, 6))
            if fig:
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"   ✅ Saved: {filename}")

        # Save combined plot
        filename = f"{prefix}_all.{format}"
        fig, _ = self.easyplot(figsize=(12, 6))
        if fig:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"   ✅ Saved: {filename}")

        print(f"✅ All plots saved successfully!")


def create_plotter(sim: EmbedSim) -> PlotHelper:
    """
    Create a PlotHelper instance for the given simulation.

    Parameters
    ----------
    sim : EmbedSim
        The simulation object containing scope data

    Returns
    -------
    PlotHelper
        Helper object for plotting

    Examples
    --------
    >>> from plot_helper import create_plotter
    >>> # Assume 'sim' is an EmbedSim instance with data
    >>> plotter = create_plotter(sim)  # doctest: +SKIP
    >>> plotter.easyplot('clarke')  # doctest: +SKIP
    """
    return PlotHelper(sim)


# Export public interface
__all__ = [
    'PlotHelper',
    'create_plotter'
]