def compute_dynamic_thresholds(audio, sr, frame_length_samples, segment_length_samples):
    """
    Computes thresholds for energy and Zero-Crossing Rate (ZCR).

    The thresholds are determined based on the mean and standard deviation
    of the energy and ZCR within individual segments of the audio signal.
    """

    # Initialization of lists for energy thresholds and ZCR thresholds
    energy_thresholds = []
    zcr_thresholds = []

    # Iterating over the audio signal in segments
    for i in range(0, len(audio), segment_length_samples):
        # Extract the current segment from the audio signal
        segment = audio[i:i + segment_length_samples]
        
        # Calculate the number of frames in the current segment
        n_frames = max(int(len(segment) / frame_length_samples), 1)
        
        # Calculate the energy for each frame in the segment
        energy = np.array([np.sum(segment[j * frame_length_samples:(j + 1) * frame_length_samples] ** 2) for j in range(n_frames)])
        
        # Calculate the Zero-Crossing Rate for each frame in the segment
        zcr = np.array([np.sum(librosa.zero_crossings(segment[j * frame_length_samples:(j + 1) * frame_length_samples], pad=False)) for j in range(n_frames)])
        
        # Add the energy threshold to the array
        # The threshold is the mean plus the standard deviation of the energy in the segment
        energy_thresholds.append(np.mean(energy) + np.std(energy))
        
        # Add the ZCR threshold to the array
        # The threshold is the mean plus the standard deviation of ZCR in the segment
        zcr_thresholds.append(np.mean(zcr) + np.std(zcr))
    
    # Return the lists of thresholds
    return energy_thresholds, zcr_thresholds