VELOCITY = 127 

CHORDS = {
    1: [60, 64, 67],   # C major
    2: [55, 59, 62],   # G major
    3: [57, 60, 64],   # A minor
    4: [53, 57, 60],   # F major
    5: [60, 65, 68],   # C major

}

CHORD_NAMES = {1: "C", 2: "G", 3: "Am", 4: "F"}

current_chord = 0

def set_chord(fs, n):
    global current_chord

    if n == current_chord:
        return n
    
    # Stop the old chord
    for note in CHORDS.get(current_chord, []):
        fs.noteoff(0, note)

    # Start the new one
    for note in CHORDS.get(n, []):
        fs.noteon(0, note, VELOCITY)

    current_chord = n

    return n
