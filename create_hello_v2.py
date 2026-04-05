"""
Create HELLO WORLD test file with proper boundaries between letters
"""
import numpy as np
import mne

# Morse code for HELLO WORLD
# H = .... (4 dots)
# E = . (1 dot)  
# L = .-.. (dot-dash-dot-dot)
# L = .-..
# O = --- (3 dashes)
# (space)
# W = .-- (dot-dash-dash)
# O = ---
# R = .-. (dot-dash-dot)
# L = .-..
# D = -.. (dash-dot-dot)

# Each signal is 1 second, sampling rate 250Hz
sfreq = 250
t = 1.0  # 1 second per signal
n_samples = int(sfreq * t)

# Create signal patterns
def create_signal(signal_type, n_ch=3, sfreq=250, duration=1.0):
    """Create a simple signal pattern"""
    n_samples = int(sfreq * duration)
    if signal_type == 'dot':  # Left hand = higher amplitude on C3 (ch0)
        data = np.random.randn(n_ch, n_samples) * 0.3
        data[0, :] += np.sin(2 * np.pi * 10 * np.arange(n_samples) / sfreq) * 2
    elif signal_type == 'dash':  # Right hand = higher amplitude on C4 (ch2)
        data = np.random.randn(n_ch, n_samples) * 0.3
        data[2, :] += np.sin(2 * np.pi * 10 * np.arange(n_samples) / sfreq) * 2
    else:  # Rest/boundary
        data = np.random.randn(n_ch, n_samples) * 0.1
    return data

# Build the sequence
signals = []
events = []
current_sample = 0

# Event IDs
LEFT = 1    # dot
RIGHT = 2   # dash
BOUNDARY = 3  # between letters
SPACE = 4     # between words

def add_letter(letter_pattern, letter_name):
    """Add a letter with its Morse pattern"""
    global current_sample
    for sig in letter_pattern:
        signals.append(create_signal(sig))
        if sig == 'dot':
            events.append([current_sample, 0, LEFT])
        else:
            events.append([current_sample, 0, RIGHT])
        current_sample += n_samples
    # Add boundary after letter
    signals.append(create_signal('boundary'))
    events.append([current_sample, 0, BOUNDARY])
    current_sample += n_samples

# H = ....
add_letter(['dot', 'dot', 'dot', 'dot'], 'H')

# E = .
add_letter(['dot'], 'E')

# L = .-..
add_letter(['dot', 'dash', 'dot', 'dot'], 'L')

# L = .-..
add_letter(['dot', 'dash', 'dot', 'dot'], 'L')

# O = ---
add_letter(['dash', 'dash', 'dash'], 'O')

# SPACE between words
signals.append(create_signal('boundary'))
events.append([current_sample, 0, SPACE])
current_sample += n_samples

# W = .--
add_letter(['dot', 'dash', 'dash'], 'W')

# O = ---
add_letter(['dash', 'dash', 'dash'], 'O')

# R = .-.
add_letter(['dot', 'dash', 'dot'], 'R')

# L = .-..
add_letter(['dot', 'dash', 'dot', 'dot'], 'L')

# D = -..
add_letter(['dash', 'dot', 'dot'], 'D')

# Concatenate all signals
data = np.concatenate(signals, axis=1)

# Create stim channel (all zeros, will be filled with events)
stim_data = np.zeros((1, data.shape[1]))
data_with_stim = np.vstack([data, stim_data])

# Create MNE info and Raw object
info = mne.create_info(ch_names=['C3', 'Cz', 'C4', 'STI 014'], sfreq=sfreq, ch_types=['eeg', 'eeg', 'eeg', 'stim'])
raw = mne.io.RawArray(data_with_stim, info)

# Add events
events = np.array(events)
for sample, _, event_id in events:
    raw._data[3, sample] = event_id

# Save
output_path = r'C:\Users\DoubleJ\Desktop\helloworld_v2.fif'
raw.save(output_path, overwrite=True)

print(f"Created: {output_path}")
print(f"Data shape: {data.shape}")
print(f"Duration: {data.shape[1]/sfreq:.1f}s")
print(f"Events: {len(events)}")
print(f"Event types: {set(events[:, 2])}")

# Verify
print("\nEvent sequence:")
for i, (s, d, e) in enumerate(events):
    name = {1: 'L', 2: 'R', 3: '|', 4: ' '}.get(e, '?')
    print(f"{i:2d}: {name}", end=' ')
    if (i+1) % 10 == 0:
        print()
print()
