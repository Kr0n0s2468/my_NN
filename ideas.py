

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.ndimage import zoom

# Use your existing functions without modification
def load_model():
    data = np.load("model_weights.npz")
    W1_flat, b1_flat, W2_flat, b2_flat = data['W1'], data['b1'], data['W2'], data['b2']
    W1 = W1_flat.reshape((10, 784))
    b1 = b1_flat.reshape((10, 1))
    W2 = W2_flat.reshape((10, 10))
    b2 = b2_flat.reshape((10, 1))
    return W1, b1, W2, b2

# Load the model once
W1, b1, W2, b2 = load_model()

class DigitDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Drawing canvas
        self.canvas_size = 280
        self.drawing = np.zeros((self.canvas_size, self.canvas_size))
        self.img = self.ax.imshow(self.drawing, cmap='gray', vmin=0, vmax=1)
        self.ax.set_title('Draw a Digit (0-9)')
        self.ax.axis('off')
        
        # Prediction display
        self.pred_text = self.ax.text(0.5, 1.05, 'Draw a digit and click "Predict"', 
                                     transform=self.ax.transAxes, ha='center', fontsize=14)
        
        # Buttons
        self.ax_clear = plt.axes([0.15, 0.05, 0.2, 0.06])
        self.ax_predict = plt.axes([0.4, 0.05, 0.2, 0.06])
        self.ax_confidence = plt.axes([0.65, 0.05, 0.2, 0.06])
        
        self.btn_clear = Button(self.ax_clear, 'Clear')
        self.btn_predict = Button(self.ax_predict, 'Predict')
        self.btn_confidence = Button(self.ax_confidence, 'Confidence')
        
        self.btn_clear.on_clicked(self.clear_canvas)
        self.btn_predict.on_clicked(self.predict_digit)
        self.btn_confidence.on_clicked(self.show_confidence)
        
        # Brush size slider
        self.ax_brush = plt.axes([0.15, 0.12, 0.7, 0.03])
        self.brush_slider = Slider(
            ax=self.ax_brush,
            label='Brush Size',
            valmin=1,
            valmax=30,
            valinit=15,
            valstep=1
        )
        self.brush_slider.on_changed(self.update_brush)
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.drawing_flag = False
        self.brush_size = 15
        
        plt.show()
    
    def update_brush(self, val):
        self.brush_size = int(val)
    
    def on_press(self, event):
        if event.inaxes == self.ax:
            self.drawing_flag = True
            self.draw(event)
    
    def on_motion(self, event):
        if self.drawing_flag and event.inaxes == self.ax:
            self.draw(event)
    
    def on_release(self, event):
        self.drawing_flag = False
    
    def draw(self, event):
        if event.xdata is None or event.ydata is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        size = self.brush_size
        half = size // 2
        
        # Draw a circle at the cursor position
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                if (i*i + j*j) <= half*half:  # Circle equation
                    xi, yj = x + i, y + j
                    if 0 <= xi < self.canvas_size and 0 <= yj < self.canvas_size:
                        # Add with decreasing intensity toward edges
                        distance = np.sqrt(i*i + j*j)
                        intensity = max(0, 1 - distance/half)
                        self.drawing[yj, xi] = min(1, self.drawing[yj, xi] + intensity)
        
        self.img.set_data(self.drawing)
        self.fig.canvas.draw_idle()
    
    def clear_canvas(self, event=None):
        self.drawing = np.zeros((self.canvas_size, self.canvas_size))
        self.img.set_data(self.drawing)
        self.pred_text.set_text('Canvas cleared')
        self.fig.canvas.draw_idle()
    
    def preprocess_drawing(self):
        # Downscale the drawing from 280x280 to 28x28
        downscaled = zoom(self.drawing, 0.1, order=1)  # Bilinear interpolation
        
        # Invert colors (MNIST has black digits on white background)
        inverted = 1 - downscaled
        
        # Flatten and normalize
        processed = inverted.reshape((784, 1))
        return processed
    
    def predict_digit(self, event=None):
        X = self.preprocess_drawing()
        
        # Use your existing forward propagation function
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        prediction = get_predictions(A2)
        
        self.pred_text.set_text(f'Prediction: {prediction[0]}')
        self.fig.canvas.draw_idle()
        return prediction[0]
    
    def show_confidence(self, event=None):
        X = self.preprocess_drawing()
        
        # Use your existing forward propagation function
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        confidences = A2.flatten()
        
        # Create confidence visualization
        fig_conf, ax_conf = plt.subplots(figsize=(10, 4))
        bars = ax_conf.bar(range(10), confidences, color='skyblue')
        ax_conf.set_title('Prediction Confidence Levels')
        ax_conf.set_xlabel('Digit')
        ax_conf.set_ylabel('Confidence')
        ax_conf.set_ylim(0, 1)
        
        # Highlight the highest confidence bar
        max_idx = np.argmax(confidences)
        bars[max_idx].set_color('salmon')
        
        # Add confidence percentages
        for i, conf in enumerate(confidences):
            ax_conf.text(i, conf + 0.02, f'{conf:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
