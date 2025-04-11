import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dv
from typing import Tuple
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft

class PendulumTracker:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.resolution = (346, 260)  # Default DVS resolution

    def load_events(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load events with robust format handling"""
        try:
            with dv.AedatFile(self.file_path) as f:
                if 'events' not in f.names:
                    raise ValueError("No events found in file")
                
                events = []
                for packet in f['events']:
                    if hasattr(packet, 'numpy'):
                        events.append(packet.numpy())
                    else:
                        # Fixed syntax here - properly matched parentheses
                        events.append(np.array(
                            [(e.timestamp, e.x, e.y, e.polarity) 
                             for e in (packet if hasattr(packet, '__iter__') else [packet])],
                            dtype=[('timestamp', 'int64'), ('x', 'int16'), 
                                  ('y', 'int16'), ('polarity', 'bool')]
                        ))
                
                if not events:
                    raise ValueError("No events loaded")
                
                events = np.concatenate(events)
                return (events['timestamp'], events['x'], 
                        events['y'], events['polarity'])
                
        except Exception as e:
            raise RuntimeError(f"Event loading failed: {str(e)}") from e

    def create_frames(self, timestamps, x, y, polarities, time_window=20000):
        """Create normalized event frames with adaptive resolution"""
        width, height = int(x.max())+1, int(y.max())+1
        time_bins = np.arange(timestamps[0], timestamps[-1], time_window)
        frames = np.zeros((len(time_bins)-1, height, width))
        
        for i in range(len(time_bins)-1):
            mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i+1])
            hist, _, _ = np.histogram2d(
                y[mask], x[mask], 
                bins=[height, width],
                weights=polarities[mask]*2-1  # [-1, 1] range
            )
            frames[i] = hist
            
        return frames

    def extract_motion(self, frames):
        """Robust pendulum angle extraction with sign consistency"""
        # PCA on normalized frames
        flat = StandardScaler().fit_transform(frames.reshape(frames.shape[0], -1))
        pca = PCA(n_components=2).fit_transform(flat)
        angle = pca[:,0]
        
        # Sign consistency using velocity
        if np.mean(np.diff(angle)) < 0:
            angle = -angle
            
        # Adaptive smoothing
        window = min(51, len(angle)//4)
        if window % 2 == 0: window -= 1
        angle = savgol_filter(angle, window, 3)
        
        # Scale to [-π/2, π/2] range
        return (angle / np.percentile(np.abs(angle), 95)) * (np.pi/2)

class PendulumDataset(Dataset):
    def __init__(self, time, angle):
        self.time = torch.FloatTensor(time).unsqueeze(1)
        self.angle = torch.FloatTensor(angle).unsqueeze(1)
        
    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, idx):
        return self.time[idx], self.angle[idx]

class PhysicsPendulumNN(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Physical parameters with constraints
        self.g = nn.Parameter(torch.tensor(9.8).clamp(8.0, 11.0))
        self.L = nn.Parameter(torch.tensor(1.0).clamp(0.3, 3.0))
        self.damping = nn.Parameter(torch.tensor(0.05).clamp(0.01, 0.1))
        
    def forward(self, t):
        return self.net(t)
    
    def physics_residual(self, t, theta):
        t.requires_grad_(True)
        theta = self(t)
        
        # 1st derivative
        dtheta = torch.autograd.grad(theta, t, torch.ones_like(theta),
                                create_graph=True, retain_graph=True)[0]
        
        # 2nd derivative
        d2theta = torch.autograd.grad(dtheta, t, torch.ones_like(dtheta),
                                 create_graph=True, retain_graph=True)[0]
        
        return d2theta + self.damping*dtheta + (self.g/self.L)*torch.sin(theta)

def train_model(model, dataset, epochs=3000):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam([
        {'params': model.net.parameters(), 'lr': 0.001},
        {'params': [model.g, model.L, model.damping], 'lr': 0.01}
    ])
    
    for epoch in range(epochs):
        for t, theta in dataloader:
            optimizer.zero_grad()
            
            # Prediction
            theta_pred = model(t)
            
            # Loss components
            loss_data = F.mse_loss(theta_pred, theta)
            loss_phys = model.physics_residual(t, theta_pred).pow(2).mean()
            
            # Adaptive weighting
            phys_weight = min(0.01, epoch/1000)
            loss = loss_data + phys_weight * loss_phys
            
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                test_t = torch.linspace(0, float(dataset.time[-1]), 1000).unsqueeze(1)
                pred = model(test_t)
                
                # Interpolate the prediction to match the length of the dataset
                interp_pred = np.interp(dataset.time.numpy(), test_t.squeeze().numpy(), pred.squeeze().numpy())
                
                # Ensure both arrays are 1-dimensional for correlation calculation
                interp_pred = interp_pred.squeeze()
                dataset_angle = dataset.angle.squeeze().numpy()
                
                # Calculate correlation with the dataset
                corr = np.corrcoef(interp_pred, dataset_angle)[0, 1]
                
            print(f"Epoch {epoch}: Loss={loss.item():.4f} "
                  f"(Data={loss_data.item():.4f}, Phys={loss_phys.item():.4f}) "
                  f"Corr={corr:.3f}")
            print(f"  Params: g={model.g.item():.2f}, L={model.L.item():.2f}, "
                  f"damping={model.damping.item():.4f}")

def main():
    try:
        # 1. Load and process data
        tracker = PendulumTracker("/Users/joshuaj27/Desktop/Physical Pendulum Test/Simulated Single Pendulum.aedat4")
        timestamps, x, y, polarities = tracker.load_events()
        
        # Convert to seconds
        time_sec = (timestamps - timestamps[0]) / 1e6
        
        # 2. Create frames and extract motion
        frames = tracker.create_frames(timestamps, x, y, polarities)
        angle = tracker.extract_motion(frames)
        frame_times = np.linspace(0, time_sec[-1], len(angle))
        
        # 3. Train model
        dataset = PendulumDataset(frame_times, angle)
        model = PhysicsPendulumNN()
        train_model(model, dataset)
        
        # 4. Visualize results
        with torch.no_grad():
            test_t = torch.linspace(0, float(frame_times[-1]), 1000).unsqueeze(1)
            pred = model(test_t)
            
            plt.figure(figsize=(14, 6))
            plt.plot(frame_times, angle, 'o', markersize=3, alpha=0.5, label='PCA Angle')
            plt.plot(test_t.numpy(), pred.numpy(), 'r-', linewidth=2, label='PINN Fit')
            
            # Calculate physics metrics
            period = 2*np.pi*torch.sqrt(model.L/model.g).item()
            plt.title(f"Pendulum Tracking | "
                     f"g={model.g.item():.2f}m/s² | "
                     f"L={model.L.item():.2f}m | "
                     f"T={period:.2f}s | "
                     f"Damping={model.damping.item():.4f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (rad)")
            plt.legend()
            plt.grid(True)
            
            # Frequency spectrum
            plt.figure(figsize=(12, 4))
            freq = np.fft.fftfreq(len(angle), d=frame_times[1]-frame_times[0])
            fft_vals = np.abs(fft(angle))
            plt.plot(freq[:len(freq)//2], fft_vals[:len(freq)//2])
            plt.axvline(1/period, color='r', linestyle='--', 
                       label=f'Predicted: {1/period:.2f}Hz')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.legend()
            plt.show()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()