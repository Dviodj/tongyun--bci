"""Create HELLO WORLD test file with correct event IDs"""
import numpy as np
import mne

# Morse code for HELLO WORLD
# H = ...., E = ., L = .-.., L = .-.., O = ---
# (space)
# W = .--, O = ---, R = .-., L = .-.., D = -..

# Event IDs: 1=dot, 2=dash, 3=boundary, 4=space
hello = [
    1,1,1,1, 3,      # H = ....
    1, 3,            # E = .
    1,2,1,1, 3,      # L = .-..
    1,2,1,1, 3,      # L = .-..
    2,2,2, 3,         # O = ---
    4,               # space
    1,2,2, 3,         # W = .--
    2,2,2, 3,         # O = ---
    1,2,1, 3,         # R = .-.
    1,2,1,1, 3,      # L = .-..
    2,1,1, 3          # D = -..
]

print(f"Total events: {len(hello)}")
print(f"Event sequence: {hello}")

# Create synthetic EEG data
sfreq = 250
duration = 1.0  # seconds per epoch
n_channels = 4
n_epochs = len(hello)

# Create channel names
ch_names = ['C3', 'Cz', 'C4', 'STI 014']
ch_types = ['eeg', 'eeg', 'eeg', 'stim']

# Generate data
data = np.random.randn(n_channels, int(n_epochs * duration * sfreq)) * 5e-6

# Add events to stim channel
for i, event_id in enumerate(hello):
    sample_start = int(i * duration * sfreq)
    sample_end = min(sample_start + int(0.1 * sfreq), data.shape[1])
    data[3, sample_start:sample_end] = event_id

# Create MNE info
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
info.set_montage('standard_1020')

# Create Raw object
raw = mne.io.RawArray(data, info)

# Save
output_path = r'C:\Users\DoubleJ\Desktop\helloworld_v3.fif'
raw.save(output_path, overwrite=True)
print(f"Saved to: {output_path}")

# Verify
print("\nVerifying events:")
events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
print(f"Found {len(events)} events")
print(f"Event values: {events[:, 2].tolist()}")
