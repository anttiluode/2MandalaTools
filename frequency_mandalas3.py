"""
Polar Sigh Cymatics
===================
Radial frequency decomposition with cymatics-style visualization.

Unlike Cartesian FFT EQ (left-right = low-high), this uses:
- Center = DC (average brightness)
- Concentric rings = frequency bands
- Radial patterns = interference/structure

Like physical cymatics where vibrations from center create standing wave patterns.

Features:
- Radial EQ with draggable ring gains
- Cymatics simulation (wave interference)
- Animated frequency sweep
- Mandala overlay system
- Multiple visualization tabs

Author: Built for Antti's consciousness crystallography research
"""

import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time

PHI = (1 + np.sqrt(5)) / 2


class RadialEQ(tk.Canvas):
    """
    Radial frequency equalizer - drag rings to adjust band gains.
    Center = low freq, outer = high freq.
    """
    def __init__(self, parent, n_bands=8, size=300, callback=None):
        super().__init__(parent, width=size, height=size, bg='#1a1a2e', highlightthickness=0)
        self.size = size
        self.center = size // 2
        self.n_bands = n_bands
        self.callback = callback
        
        # Gains for each ring (0-1)
        self.gains = np.ones(n_bands)
        
        # Ring radii (phi-spaced)
        self.ring_radii = self._compute_phi_radii()
        
        # Interaction
        self.selected_ring = None
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        
        self.draw()
        
    def _compute_phi_radii(self):
        """Compute ring radii at golden ratio intervals."""
        max_r = self.center * 0.95
        radii = []
        r = max_r
        for i in range(self.n_bands):
            radii.append(r)
            r /= PHI ** 0.5
        return list(reversed(radii))  # Inner to outer
        
    def _on_click(self, event):
        # Find which ring was clicked
        dx = event.x - self.center
        dy = event.y - self.center
        dist = np.sqrt(dx*dx + dy*dy)
        
        for i, r in enumerate(self.ring_radii):
            if abs(dist - r) < 15:
                self.selected_ring = i
                break
                
    def _on_drag(self, event):
        if self.selected_ring is not None:
            dx = event.x - self.center
            dy = event.y - self.center
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Map distance to gain
            target_r = self.ring_radii[self.selected_ring]
            # Closer to center = lower gain, further = higher
            gain = np.clip(dist / target_r, 0.1, 2.0)
            self.gains[self.selected_ring] = gain
            
            self.draw()
            if self.callback:
                self.callback()
                
    def _on_release(self, event):
        self.selected_ring = None
        
    def draw(self):
        self.delete("all")
        
        # Draw rings
        for i, r in enumerate(self.ring_radii):
            gain = self.gains[i]
            
            # Color based on gain
            intensity = int(min(255, gain * 128))
            color = f'#{intensity:02x}{intensity//2:02x}{255-intensity:02x}'
            
            # Ring thickness based on gain
            width = max(1, int(gain * 4))
            
            self.create_oval(
                self.center - r, self.center - r,
                self.center + r, self.center + r,
                outline=color, width=width
            )
            
            # Gain label
            angle = -np.pi/4  # 45 degrees
            lx = self.center + r * np.cos(angle) * 0.7
            ly = self.center + r * np.sin(angle) * 0.7
            self.create_text(lx, ly, text=f'{gain:.1f}', fill='white', font=('Arial', 8))
            
        # Center dot
        self.create_oval(
            self.center-5, self.center-5,
            self.center+5, self.center+5,
            fill='#ff6b6b', outline=''
        )
        
        # Labels
        self.create_text(self.center, 20, text='HIGH FREQ', fill='#666', font=('Arial', 9))
        self.create_text(self.center, self.size-20, text='(drag rings to adjust)', fill='#444', font=('Arial', 8))
        
    def get_radial_filter(self, size):
        """Generate 2D radial filter from ring gains."""
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        # Interpolate gains across radii
        ring_norm = np.array(self.ring_radii) / (self.center * 0.95)
        
        # Create filter by interpolating gains
        filt = np.interp(dist_norm, ring_norm, self.gains)
        
        return filt.astype(np.float32)


class CymaticsSimulator:
    """
    Simulates cymatics-style wave interference patterns.
    """
    def __init__(self, size=512):
        self.size = size
        self.time = 0
        self.frequencies = [1, PHI, PHI**2, PHI**3]  # Phi-harmonic series
        self.amplitudes = [1.0, 0.5, 0.25, 0.125]
        self.damping = 0.98
        
        # Wave field
        self.field = np.zeros((size, size), dtype=np.float32)
        self.velocity = np.zeros((size, size), dtype=np.float32)
        
        # Coordinate grids
        cy, cx = size // 2, size // 2
        y, x = np.mgrid[:size, :size]
        self.r = np.sqrt((x - cx)**2 + (y - cy)**2) / (size / 2)
        self.theta = np.arctan2(y - cy, x - cx)
        
    def step(self, dt=0.1, source_freq=None):
        """Advance simulation one step."""
        self.time += dt
        
        # Source excitation at center
        if source_freq is None:
            # Multi-frequency source
            source = np.zeros((self.size, self.size), dtype=np.float32)
            for f, a in zip(self.frequencies, self.amplitudes):
                source += a * np.sin(2 * np.pi * f * self.time) * np.exp(-self.r * 10)
        else:
            source = np.sin(2 * np.pi * source_freq * self.time) * np.exp(-self.r * 10)
            
        # Wave equation update (simplified)
        laplacian = cv2.Laplacian(self.field, cv2.CV_32F)
        self.velocity += laplacian * 0.1 + source * 0.5
        self.velocity *= self.damping
        self.field += self.velocity * dt
        
        # Boundary absorption
        boundary = np.exp(-((1 - self.r) * 5) ** 2)
        self.field *= boundary
        
        return self.field
        
    def get_standing_wave(self, n_nodes=5, symmetry=8):
        """Generate standing wave pattern (Chladni-like)."""
        pattern = np.zeros((self.size, self.size), dtype=np.float32)
        
        for n in range(1, n_nodes + 1):
            for m in range(symmetry):
                angle_offset = m * 2 * np.pi / symmetry
                # Bessel-like radial + angular modulation
                radial = np.sin(n * np.pi * self.r)
                angular = np.cos(symmetry * self.theta + angle_offset + self.time * n)
                pattern += radial * angular / n
                
        # Normalize
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
        
        return pattern


class PolarSighCymatics:
    """
    Main application - Polar frequency decomposition with cymatics visualization.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Polar Sigh Cymatics")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f0f1a')
        
        # State
        self.original_image = None
        self.current_image = None
        self.spectrum = None
        self.filtered_bands = []
        self.cymatics_sim = CymaticsSimulator(256)
        
        # Animation
        self.animating = False
        self.animation_phase = 0
        self.sweep_mode = 'none'  # 'none', 'sweep', 'pulse', 'cymatics'
        
        self.setup_gui()
        self.start_animation_loop()
        
    def setup_gui(self):
        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Load button
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(pady=10)
        
        # Radial EQ
        eq_frame = ttk.LabelFrame(left_panel, text="Radial Frequency EQ")
        eq_frame.pack(fill=tk.X, pady=10)
        
        self.radial_eq = RadialEQ(eq_frame, n_bands=8, size=300, callback=self.on_eq_change)
        self.radial_eq.pack(pady=10)
        
        # Presets
        preset_frame = ttk.Frame(left_panel)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="Low", command=self.preset_low).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Mid", command=self.preset_mid).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="High", command=self.preset_high).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="All", command=self.preset_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Phi", command=self.preset_phi).pack(side=tk.LEFT, padx=2)
        
        # Animation controls
        anim_frame = ttk.LabelFrame(left_panel, text="Animation")
        anim_frame.pack(fill=tk.X, pady=10)
        
        self.sweep_var = tk.StringVar(value='none')
        ttk.Radiobutton(anim_frame, text="None", variable=self.sweep_var, 
                       value='none', command=self.on_sweep_change).pack(anchor=tk.W)
        ttk.Radiobutton(anim_frame, text="Frequency Sweep", variable=self.sweep_var,
                       value='sweep', command=self.on_sweep_change).pack(anchor=tk.W)
        ttk.Radiobutton(anim_frame, text="Pulse", variable=self.sweep_var,
                       value='pulse', command=self.on_sweep_change).pack(anchor=tk.W)
        ttk.Radiobutton(anim_frame, text="Cymatics Sim", variable=self.sweep_var,
                       value='cymatics', command=self.on_sweep_change).pack(anchor=tk.W)
        
        # Symmetry control
        sym_frame = ttk.Frame(anim_frame)
        sym_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sym_frame, text="Symmetry:").pack(side=tk.LEFT)
        self.symmetry_var = tk.IntVar(value=8)
        sym_spin = ttk.Spinbox(sym_frame, from_=1, to=16, width=5, 
                               textvariable=self.symmetry_var, command=self.on_eq_change)
        sym_spin.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = ttk.Frame(anim_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.05)
        speed_scale = ttk.Scale(speed_frame, from_=0.01, to=0.2, 
                               variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Right panel - Notebook with tabs
        right_panel = ttk.Frame(main)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Original + Filtered
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Filtered")
        
        tab1_top = ttk.Frame(tab1)
        tab1_top.pack(fill=tk.BOTH, expand=True)
        
        self.original_label = ttk.Label(tab1_top, text="Original")
        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.filtered_label = ttk.Label(tab1_top, text="Filtered")
        self.filtered_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Tab 2: Frequency Bands
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Bands")
        
        self.bands_label = ttk.Label(tab2)
        self.bands_label.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Cymatics
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Cymatics")
        
        self.cymatics_label = ttk.Label(tab3)
        self.cymatics_label.pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Mandala Overlay
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text="Mandala")
        
        self.mandala_label = ttk.Label(tab4)
        self.mandala_label.pack(fill=tk.BOTH, expand=True)
        
        # Tab 5: Spectrum
        tab5 = ttk.Frame(self.notebook)
        self.notebook.add(tab5, text="Spectrum")
        
        self.spectrum_label = ttk.Label(tab5)
        self.spectrum_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin")
        ttk.Label(self.root, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")]
        )
        if not path:
            return
            
        img = cv2.imread(path)
        if img is None:
            self.status_var.set(f"Failed to load: {path}")
            return
            
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Make square and resize
        h, w = gray.shape
        size = min(h, w, 512)
        start_y, start_x = (h - size) // 2, (w - size) // 2
        gray = gray[start_y:start_y+size, start_x:start_x+size]
        gray = cv2.resize(gray, (512, 512))
        
        self.original_image = gray.astype(np.float32) / 255.0
        
        # Compute FFT
        self.spectrum = fftshift(fft2(self.original_image))
        
        self.status_var.set(f"Loaded: {path.split('/')[-1]} (512x512)")
        self.process_and_display()
        
    def process_and_display(self):
        if self.original_image is None:
            return
            
        size = self.original_image.shape[0]
        
        # Get radial filter
        radial_filter = self.radial_eq.get_radial_filter(size)
        
        # Apply filter to spectrum
        filtered_spectrum = self.spectrum * radial_filter
        
        # Inverse FFT
        filtered = np.real(ifft2(ifftshift(filtered_spectrum)))
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-10)
        
        self.current_image = filtered
        
        # Update displays
        self.update_filtered_tab()
        self.update_bands_tab()
        self.update_spectrum_tab()
        self.update_mandala_tab()
        
    def update_filtered_tab(self):
        if self.original_image is None:
            return
            
        # Original
        orig_display = (self.original_image * 255).astype(np.uint8)
        orig_pil = Image.fromarray(orig_display).resize((400, 400))
        self.orig_tk = ImageTk.PhotoImage(orig_pil)
        self.original_label.configure(image=self.orig_tk)
        
        # Filtered with colormap
        if self.current_image is not None:
            filt_u8 = (self.current_image * 255).astype(np.uint8)
            filt_color = cv2.applyColorMap(filt_u8, cv2.COLORMAP_INFERNO)
            filt_rgb = cv2.cvtColor(filt_color, cv2.COLOR_BGR2RGB)
            filt_pil = Image.fromarray(filt_rgb).resize((400, 400))
            self.filt_tk = ImageTk.PhotoImage(filt_pil)
            self.filtered_label.configure(image=self.filt_tk)
            
    def update_bands_tab(self):
        if self.spectrum is None:
            return
            
        size = 512
        n_bands = self.radial_eq.n_bands
        band_size = 128
        
        # Extract each band separately
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        ring_radii = np.array(self.radial_eq.ring_radii) / (self.radial_eq.center * 0.95)
        
        # Create grid of bands
        cols = 4
        rows = (n_bands + cols - 1) // cols
        grid = np.zeros((rows * band_size, cols * band_size, 3), dtype=np.uint8)
        
        self.filtered_bands = []
        
        for i in range(n_bands):
            # Create band mask
            if i == 0:
                inner = 0
            else:
                inner = ring_radii[i-1]
            outer = ring_radii[i]
            
            mask = ((dist_norm >= inner) & (dist_norm < outer)).astype(np.float32)
            
            # Smooth edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Extract band
            band_spectrum = self.spectrum * mask * self.radial_eq.gains[i]
            band_spatial = np.real(ifft2(ifftshift(band_spectrum)))
            
            # Normalize
            band_spatial = np.abs(band_spatial)
            if band_spatial.max() > 0:
                band_spatial = band_spatial / band_spatial.max()
                
            self.filtered_bands.append(band_spatial)
            
            # Colorize and resize
            band_u8 = (band_spatial * 255).astype(np.uint8)
            band_color = cv2.applyColorMap(band_u8, cv2.COLORMAP_INFERNO)
            band_small = cv2.resize(band_color, (band_size, band_size))
            
            # Add to grid
            row, col = i // cols, i % cols
            grid[row*band_size:(row+1)*band_size, col*band_size:(col+1)*band_size] = band_small
            
            # Label
            cv2.putText(grid, f'Band {i+1}', (col*band_size+5, row*band_size+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        # Display
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        grid_pil = Image.fromarray(grid_rgb)
        self.bands_tk = ImageTk.PhotoImage(grid_pil)
        self.bands_label.configure(image=self.bands_tk)
        
    def update_spectrum_tab(self):
        if self.spectrum is None:
            return
            
        # Magnitude spectrum with filter overlay
        magnitude = np.log1p(np.abs(self.spectrum))
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        
        # Get filter shape
        radial_filter = self.radial_eq.get_radial_filter(512)
        
        # Combine: spectrum in blue/purple, filter in orange
        display = np.zeros((512, 512, 3), dtype=np.float32)
        display[:, :, 0] = magnitude * 0.5  # Red
        display[:, :, 1] = magnitude * radial_filter  # Green (filtered parts)
        display[:, :, 2] = magnitude  # Blue
        
        display = (np.clip(display, 0, 1) * 255).astype(np.uint8)
        
        # Draw ring positions
        for r in self.radial_eq.ring_radii:
            r_pixels = int(r / (self.radial_eq.center * 0.95) * 256)
            cv2.circle(display, (256, 256), r_pixels, (255, 255, 0), 1)
            
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        spec_pil = Image.fromarray(display_rgb).resize((500, 500))
        self.spec_tk = ImageTk.PhotoImage(spec_pil)
        self.spectrum_label.configure(image=self.spec_tk)
        
    def update_mandala_tab(self):
        if self.current_image is None:
            return
            
        # Create mandala from filtered image
        mandala = self.create_mandala(self.current_image, self.symmetry_var.get())
        
        # Colorize
        mandala_u8 = (mandala * 255).astype(np.uint8)
        mandala_color = cv2.applyColorMap(mandala_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        mandala_rgb = cv2.cvtColor(mandala_color, cv2.COLOR_BGR2RGB)
        mandala_pil = Image.fromarray(mandala_rgb).resize((600, 600))
        self.mandala_tk = ImageTk.PhotoImage(mandala_pil)
        self.mandala_label.configure(image=self.mandala_tk)
        
    def update_cymatics_tab(self):
        # Get cymatics pattern
        if self.sweep_mode == 'cymatics':
            pattern = self.cymatics_sim.step(dt=self.speed_var.get())
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
        else:
            pattern = self.cymatics_sim.get_standing_wave(
                n_nodes=5, 
                symmetry=self.symmetry_var.get()
            )
            
        # Apply symmetry
        mandala = self.create_mandala(pattern, self.symmetry_var.get())
        
        # If we have an image, blend with cymatics
        if self.current_image is not None:
            # Resize cymatics to match
            cym_resized = cv2.resize(mandala, (self.current_image.shape[1], self.current_image.shape[0]))
            
            # Blend
            blend = self.current_image * 0.5 + cym_resized * 0.5
            blend = (blend - blend.min()) / (blend.max() - blend.min() + 1e-10)
        else:
            blend = mandala
            
        # Colorize
        blend_u8 = (blend * 255).astype(np.uint8)
        blend_color = cv2.applyColorMap(blend_u8, cv2.COLORMAP_INFERNO)
        
        blend_rgb = cv2.cvtColor(blend_color, cv2.COLOR_BGR2RGB)
        blend_pil = Image.fromarray(blend_rgb).resize((600, 600))
        self.cym_tk = ImageTk.PhotoImage(blend_pil)
        self.cymatics_label.configure(image=self.cym_tk)
        
    def create_mandala(self, img, symmetry):
        """Apply kaleidoscope symmetry to create mandala."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        result = np.zeros_like(img, dtype=np.float32)
        
        for i in range(symmetry):
            angle = i * 360.0 / symmetry
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img.astype(np.float32), M, (w, h))
            
            if i % 2 == 1:
                rotated = cv2.flip(rotated, 1)
                
            result = np.maximum(result, rotated)
            
        return result
        
    def on_eq_change(self):
        self.process_and_display()
        
    def on_sweep_change(self):
        self.sweep_mode = self.sweep_var.get()
        
    def preset_low(self):
        self.radial_eq.gains = np.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])
        self.radial_eq.draw()
        self.process_and_display()
        
    def preset_mid(self):
        self.radial_eq.gains = np.array([0.2, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, 0.2])
        self.radial_eq.draw()
        self.process_and_display()
        
    def preset_high(self):
        self.radial_eq.gains = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
        self.radial_eq.draw()
        self.process_and_display()
        
    def preset_all(self):
        self.radial_eq.gains = np.ones(8)
        self.radial_eq.draw()
        self.process_and_display()
        
    def preset_phi(self):
        # Gains following phi pattern
        self.radial_eq.gains = np.array([1/PHI**3, 1/PHI**2, 1/PHI, 1, PHI, PHI, 1/PHI, 1/PHI**2])
        self.radial_eq.draw()
        self.process_and_display()
        
    def start_animation_loop(self):
        """Start background animation thread."""
        self.animating = True
        self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.animation_thread.start()
        
    def _animation_loop(self):
        """Background animation loop."""
        while self.animating:
            if self.sweep_mode == 'none':
                time.sleep(0.1)
                continue
                
            self.animation_phase += self.speed_var.get()
            
            if self.sweep_mode == 'sweep':
                # Sweep through frequencies
                sweep_idx = int(self.animation_phase * 2) % self.radial_eq.n_bands
                for i in range(self.radial_eq.n_bands):
                    dist = abs(i - sweep_idx)
                    self.radial_eq.gains[i] = max(0.1, 1.5 - dist * 0.3)
                    
            elif self.sweep_mode == 'pulse':
                # Pulsing rings
                for i in range(self.radial_eq.n_bands):
                    phase_offset = i * 0.3
                    self.radial_eq.gains[i] = 0.5 + 0.5 * np.sin(self.animation_phase * 3 + phase_offset)
                    
            # Update display (must be in main thread)
            try:
                self.root.after(0, self._update_animation)
            except:
                pass
                
            time.sleep(0.05)
            
    def _update_animation(self):
        """Update display from animation (called in main thread)."""
        self.radial_eq.draw()
        
        if self.sweep_mode in ['sweep', 'pulse']:
            self.process_and_display()
        
        # Always update cymatics tab
        self.update_cymatics_tab()
        
    def on_close(self):
        self.animating = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PolarSighCymatics(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()