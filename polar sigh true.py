"""
Polar Sigh True v2
==================
Radial frequency decomposition with "Holographic" low-res processing.

Replicates the 'Sigh Image' aesthetic by processing FFT at lower resolutions
(mimicking the 64x64 grid of the original sigh system) to reveal 
standing wave interference patterns (Moiré) in the mid-bands.

Features:
- Resolution Control: Lower resolution = stronger Moiré/Wave patterns.
- Mandala Engine: Spinds the wave patterns into high-res symmetries.
- Band Isolation: View specific frequency rings.

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

class PolarSighTrue:
    """
    True polar frequency decomposition with visible spatial patterns per band.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Polar Sigh True v2 - Holographic Interference")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0a12')
        
        # Image state
        self.original_path = None
        self.original_high_res = None # The full res image for reference
        self.process_buffer = None    # The low-res buffer for FFT (creates the moire)
        self.spectrum = None
        
        # Settings
        self.display_size = 512       # Size of the viewports
        self.process_size = 128       # Internal FFT resolution (Lower = more 'wave' like)
        self.n_bands = 10
        
        # Band gains
        self.gains = np.ones(self.n_bands)
        
        # Animation
        self.animating = False
        self.anim_phase = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main layout
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Controls
        left = ttk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left.pack_propagate(False)
        
        # --- Header / Load ---
        ttk.Button(left, text="Load Image", command=self.load_image).pack(pady=10, fill=tk.X)
        
        # --- Resolution Control ---
        res_frame = ttk.LabelFrame(left, text="Holographic Resolution")
        res_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(res_frame, text="Internal Grid (Lower = More Moiré):").pack(anchor=tk.W)
        self.res_var = tk.IntVar(value=128)
        res_combo = ttk.Combobox(res_frame, textvariable=self.res_var, 
                                values=[64, 96, 128, 256, 512], state="readonly")
        res_combo.pack(fill=tk.X, pady=2)
        res_combo.bind("<<ComboboxSelected>>", self.on_res_change)
        
        # --- Band Sliders ---
        sliders_frame = ttk.LabelFrame(left, text="Frequency Bands (Low → High)")
        sliders_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.sliders = []
        band_names = ['DC/Glow', 'Low 1', 'Low 2', 'Mid-Low', 'MID (Waves)', 
                      'MID (Interf)', 'Mid-High', 'High 1', 'High 2', 'Nyquist']
        
        for i in range(self.n_bands):
            frame = ttk.Frame(sliders_frame)
            frame.pack(fill=tk.X, pady=1)
            
            # Label
            lbl = ttk.Label(frame, text=f"{i+1}. {band_names[i]}", width=12, font=('Consolas', 8))
            lbl.pack(side=tk.LEFT)
            
            # Slider
            var = tk.DoubleVar(value=1.0)
            slider = ttk.Scale(frame, from_=0, to=2, variable=var, 
                              orient=tk.HORIZONTAL, command=lambda v, idx=i: self.on_slider_change(idx))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Value label
            val_lbl = ttk.Label(frame, text="1.0", width=4, font=('Consolas', 8))
            val_lbl.pack(side=tk.RIGHT)
            
            self.sliders.append((var, val_lbl))
            
        # --- Presets ---
        preset_frame = ttk.LabelFrame(left, text="Presets")
        preset_frame.pack(fill=tk.X, pady=10)
        
        presets = [
            ("All", self.preset_all),
            ("Low", self.preset_low),
            ("Moiré/Mid", self.preset_mid),
            ("High", self.preset_high),
            ("No DC", self.preset_no_dc),
            ("Phi", self.preset_phi),
        ]
        
        for i, (name, cmd) in enumerate(presets):
            btn = ttk.Button(preset_frame, text=name, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky='ew')
            
        # --- Animation ---
        anim_frame = ttk.LabelFrame(left, text="Animation")
        anim_frame.pack(fill=tk.X, pady=10)
        
        self.anim_var = tk.StringVar(value='none')
        for mode in ['None', 'Sweep', 'Pulse', 'Wave']:
            ttk.Radiobutton(anim_frame, text=mode, variable=self.anim_var,
                           value=mode.lower(),
                           command=self.on_anim_change).pack(anchor=tk.W)
        
        # Symmetry
        sym_frame = ttk.Frame(left)
        sym_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sym_frame, text="Mandala Symmetry:").pack(side=tk.LEFT)
        self.sym_var = tk.IntVar(value=6)
        ttk.Spinbox(sym_frame, from_=1, to=55000, textvariable=self.sym_var,
                   width=5, command=self.update_display).pack(side=tk.RIGHT)
        
        # Right: Display area
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self.tabs = {}
        for name in ["Bands", "Filtered", "Mandala", "Spectrum"]:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=name)
            canvas = tk.Canvas(tab, bg='#0a0a12', highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            self.tabs[name] = canvas
        
        self.status_var = tk.StringVar(value="Load an image to begin")
        ttk.Label(self.root, textvariable=self.status_var, background='#0a0a12', foreground='gray').pack(side=tk.BOTTOM, fill=tk.X)
        
        self.start_animation()
        
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")]
        )
        if not path:
            return
            
        self.original_path = path
        self.process_image()
        
    def on_res_change(self, event=None):
        """Update processing resolution."""
        self.process_size = self.res_var.get()
        if self.original_path:
            self.process_image()
            
    def process_image(self):
        """Reload and re-process the image at current resolution settings."""
        if not self.original_path:
            return
            
        img = cv2.imread(self.original_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
            
        # 1. Prepare High Res (for Reference)
        h, w = img.shape
        crop_size = min(h, w)
        cy, cx = h//2, w//2
        half = crop_size // 2
        img_crop = img[cy-half:cy+half, cx-half:cx+half]
        
        self.original_high_res = cv2.resize(img_crop, (self.display_size, self.display_size)).astype(np.float32) / 255.0
        
        # 2. Prepare Low Res (for Holographic Physics/Interference)
        # We resize down to create the "Sigh" effect (moire/waves)
        self.process_buffer = cv2.resize(img_crop, (self.process_size, self.process_size)).astype(np.float32) / 255.0
        
        # 3. Compute FFT on the LOW RES buffer
        self.spectrum = fftshift(fft2(self.process_buffer))
        
        # 4. Build frequency grid for low res
        cy, cx = self.process_size//2, self.process_size//2
        y, x = np.ogrid[:self.process_size, :self.process_size]
        dist = np.sqrt((x-cx)**2 + (y-cy)**2)
        # Normalize 0 to 1 (Nyquist)
        self.freq_grid = dist / (self.process_size * np.sqrt(2) / 2)
        
        self.status_var.set(f"Loaded: {self.original_path.split('/')[-1]} | Grid: {self.process_size}x{self.process_size}")
        self.update_display()
        
    def get_band_boundaries(self):
        # Non-linear spacing to emphasize mid-range interference
        return [0, 0.02, 0.05, 0.10, 0.16, 0.24, 0.35, 0.50, 0.70, 0.85, 1.0][:self.n_bands+1]
        
    def extract_band(self, band_idx):
        if self.spectrum is None:
            return np.zeros((self.process_size, self.process_size))
            
        bounds = self.get_band_boundaries()
        low = bounds[band_idx]
        high = bounds[band_idx + 1]
        
        # Create Ring Mask
        # Sigmoid smoothing for organic boundaries
        k = 30 # Steepness
        mask_inner = 1 / (1 + np.exp(-k * (self.freq_grid - low)))
        mask_outer = 1 / (1 + np.exp(-k * (high - self.freq_grid)))
        mask = mask_inner * mask_outer
        
        # Apply Gain
        band_spectrum = self.spectrum * mask * self.gains[band_idx]
        
        # Inverse FFT to get spatial waves
        spatial = np.real(ifft2(ifftshift(band_spectrum)))
        
        return spatial
        
    def extract_all_bands(self):
        return [self.extract_band(i) for i in range(self.n_bands)]
        
    def apply_mandala(self, img, symmetry):
        if symmetry <= 1:
            return img
        
        # Ensure we are working with a square image
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        # Create accumulation buffer
        result = np.zeros_like(img, dtype=np.float32)
        
        # Create base wedge mask (optional, but rotation max usually works well enough)
        
        for i in range(symmetry):
            angle = i * 360.0 / symmetry
            
            # Rotate
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Mirror logic (Kaleidoscope style)
            if i % 2 == 1:
                rotated = cv2.flip(rotated, 1)
                # Re-rotate after flip to align
                M_flip = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(rotated, M_flip, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Blend: Use Max to keep bright features (crystals)
            result = np.maximum(result, rotated)
            
        return result

    def normalize_for_display(self, img):
        # Normalize with a bit of contrast stretch for the "Sigh" glow
        if img.max() == img.min():
            return img
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def upsample(self, img, target_size):
        # BICUBIC is important here to make the low-res waves look like 
        # smooth interference patterns rather than pixels
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    def update_display(self, *args):
        if self.spectrum is None:
            return
            
        bands_spatial_lowres = self.extract_all_bands()
        
        # 1. Update Band Grid
        self.draw_bands_grid(bands_spatial_lowres)
        
        # 2. Composite (Sum bands)
        composite_lowres = np.sum(bands_spatial_lowres, axis=0)
        
        # Upscale for main display
        composite_highres = self.upsample(composite_lowres, self.display_size)
        composite_norm = self.normalize_for_display(composite_highres)
        
        # 3. Update Filtered Tab
        self.draw_filtered(composite_norm)
        
        # 4. Update Mandala
        self.draw_mandala(composite_norm)
        
        # 5. Spectrum
        self.draw_spectrum()
        
    def draw_bands_grid(self, bands):
        cols, rows = 5, 2
        w, h = 240, 240 # Thumbnail size
        full_w, full_h = cols * w, rows * h
        canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        
        bounds = self.get_band_boundaries()
        
        for i, band_lr in enumerate(bands):
            r, c = i // cols, i % cols
            
            # Upscale individual band for display so it looks smooth
            band_hr = self.upsample(band_lr, w)
            
            # Normalize - Center zero
            # We want waves: negative = blue/black, positive = bright/color
            vis = (band_hr - band_hr.mean()) / (np.std(band_hr) * 4 + 1e-6) + 0.5
            vis = np.clip(vis, 0, 1)
            
            vis_u8 = (vis * 255).astype(np.uint8)
            # Twilight is great for wave interference (cyclic colors)
            vis_color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TWILIGHT)
            
            y1, y2 = r*h, (r+1)*h
            x1, x2 = c*w, (c+1)*w
            
            canvas[y1:y2, x1:x2] = vis_color
            
            # Overlay Info
            label = f"Band {i+1}: {bounds[i]:.2f}-{bounds[i+1]:.2f}"
            cv2.putText(canvas, label, (x1+10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(canvas, f"x{self.gains[i]:.1f}", (x1+10, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        self.set_canvas_image("Bands", canvas)

    def draw_filtered(self, composite):
        # Side by side: Original (High Res) vs Filtered (Upscaled Low Res)
        
        # Left: Original
        orig_u8 = (self.original_high_res * 255).astype(np.uint8)
        orig_color = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)
        
        # Right: Filtered (with Inferno for heat/intensity look)
        filt_u8 = (composite * 255).astype(np.uint8)
        filt_color = cv2.applyColorMap(filt_u8, cv2.COLORMAP_INFERNO)
        
        # Combine
        display = np.hstack([orig_color, filt_color])
        
        # Text
        h, w = self.display_size, self.display_size
        cv2.putText(display, "SOURCE (High Res)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display, f"HOLOGRAPHIC (Grid: {self.process_size}px)", (w+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        self.set_canvas_image("Filtered", display)
        
    def draw_mandala(self, composite):
        # 1. Apply Symmetry to the wave pattern
        mandala = self.apply_mandala(composite, self.sym_var.get())
        
        # 2. Colorize
        # Twilight Shifted is great for crystals/magic look
        man_u8 = (np.clip(mandala, 0, 1) * 255).astype(np.uint8)
        man_color = cv2.applyColorMap(man_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        # 3. Post-process sharpen to make it "Crystalline"
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        man_sharp = cv2.filter2D(man_color, -1, kernel)
        
        # Center display
        final = cv2.resize(man_sharp, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        
        self.set_canvas_image("Mandala", final)
        
    def draw_spectrum(self):
        # Visualize the spectrum magnitude with rings
        mag = np.log1p(np.abs(self.spectrum))
        mag = (mag - mag.min()) / (mag.max() - mag.min())
        mag_u8 = (mag * 255).astype(np.uint8)
        mag_color = cv2.applyColorMap(mag_u8, cv2.COLORMAP_VIRIDIS)
        
        # Resize to display size
        disp = cv2.resize(mag_color, (500, 500), interpolation=cv2.INTER_NEAREST)
        
        # Draw active rings
        center = 250
        max_r = int(np.sqrt(2) * center)
        bounds = self.get_band_boundaries()
        
        for i, b in enumerate(bounds[:-1]):
            r = int(b * center) # Radius relative to side
            # Color is bright if band is active, dim if not
            gain = self.gains[i] if i < len(self.gains) else 0
            val = int(min(gain * 120, 255))
            color = (0, val, val)
            cv2.circle(disp, (center, center), r, color, 1)
            
        self.set_canvas_image("Spectrum", disp)

    def set_canvas_image(self, tab_name, numpy_img):
        # Helper to convert cv2 -> tkinter
        rgb = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas = self.tabs[tab_name]
        canvas.delete("all")
        
        # Center image
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10: cw, ch = 800, 800 # Fallback if not mapped yet
        
        canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=tk_img)
        canvas.image = tk_img # Keep ref

    def on_slider_change(self, idx):
        self.gains[idx] = self.sliders[idx][0].get()
        self.sliders[idx][1].config(text=f"{self.gains[idx]:.1f}")
        self.update_display()
        
    def on_anim_change(self):
        pass # Thread handles logic

    # --- Presets ---
    def set_gains(self, vals):
        for i, v in enumerate(vals):
            if i < len(self.sliders):
                self.sliders[i][0].set(v)
                self.gains[i] = v
                self.sliders[i][1].config(text=f"{v:.1f}")
        self.update_display()

    def preset_all(self): self.set_gains([1.0]*10)
    def preset_low(self): self.set_gains([2.0, 1.5, 1.0, 0.5, 0.1, 0, 0, 0, 0, 0])
    # The "Moire" preset focuses on the mid-bands where the 128px grid creates patterns
    def preset_mid(self): self.set_gains([0, 0, 0.2, 1.5, 2.0, 2.0, 1.5, 0.2, 0, 0])
    def preset_high(self): self.set_gains([0, 0, 0, 0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.0])
    def preset_no_dc(self): self.set_gains([0.0] + [1.0]*9)
    def preset_phi(self): self.set_gains([1/PHI**3, 1/PHI**2, 1/PHI, 1.0, PHI, PHI, 1.0, 1/PHI, 1/PHI**2, 0])

    def start_animation(self):
        self.animating = True
        threading.Thread(target=self._anim_loop, daemon=True).start()
        
    def _anim_loop(self):
        while self.animating:
            mode = self.anim_var.get()
            if mode != 'none':
                self.anim_phase += 0.1
                
                if mode == 'sweep':
                    # Moving bandpass
                    center = (np.sin(self.anim_phase)*0.5 + 0.5) * self.n_bands
                    for i in range(self.n_bands):
                        dist = abs(i - center)
                        val = np.exp(-dist*dist) * 2.0
                        self.gains[i] = val
                
                elif mode == 'pulse':
                    val = 1.0 + np.sin(self.anim_phase) * 0.5
                    self.gains[:] = val
                    
                self.root.after(0, lambda: self.set_gains(self.gains))
                
            time.sleep(0.05)
            
    def on_close(self):
        self.animating = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PolarSighTrue(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()