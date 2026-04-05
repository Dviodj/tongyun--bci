"""Create HELLO WORLD test file with correct event timing"""
import numpy as np
import mne

# Morse code for HELLO WORLD
# H = ...., E = ., L = .-.., L = .-.., O = ---
# (space)
# W = .--, O = ---, R = .-., L = .-.., D = -..

# Event IDs: 1=dot, 2=dash, 3=boundary, 4=space
# Each letter: signals + boundary
hello_world = [
    # H = ....
    1, 1, 1, 1, 3,
    # E = .
    1, 3,
    # L = .-..
    1, 2, 1, 1, 3,
    # L = .-..
    1, 2, 1, 1, 3,
    # O = ---
    2, 2, 2, 3,
    # space
    4,
    # W = .--
    1, 2, 2, 3,
    # O = ---
    2, 2, 2, 3,
    # R = .-.
    1, 2, 1, 3,
    # L = .-..
    1, 2, 1, 1, 3,
    # D = -..
    2, 1, 1, 3
]

print(f"Total events: {len(hello_world)}")

# Create synthetic EEG data
sfreq = 250
duration = 0.5  # seconds per epoch (faster)
n_channels = 4
n_epochs = len(hello_world)

# Create channel names
ch_names = ['C3', 'Cz', 'C4', 'STI 014']
ch_types = ['eeg', 'eeg', 'eeg', 'stim']

# Generate data
data = np.random.randn(n_channels, int(n_epochs * duration * sfreq)) * 5e-6

# Ensure stim channel starts at 0
data[3, :] = 0

# Add events to stim channel (start at sample 10 to avoid initial offset issue)
for i, event_id in enumerate(hello_world):
    sample_start = 10 + int(i * duration * sfreq)
    sample_end = min(sample_start + int(0.05 * sfreq), data.shape[1])
    if sample_start < data.shape[1]:
        data[3, sample_start:sample_end] = event_id

# Create MNE info
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
info.set_montage('standard_1020')

# Create Raw object
raw = mne.io.RawArray(data, info)

# Save
output_path = r'C:\Users\DoubleJ\Desktop\helloworld_v4.fif'
raw.save(output_path, overwrite=True)
print(f"Saved to: {output_path}")

# Verify
print("\nVerifying events:")
events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
print(f"Found {len(events)} events")
print(f"Event values: {events[:, 2].tolist()}")

# Decode to verify
morse = {1: '.', 2: '-', 3: '|', 4: ' '}
decoded = ''.join([morse.get(e, '?') for e in events[:, 2]])
print(f"\nDecoded: {decoded}")
