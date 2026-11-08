import math

def fingers_up(lm):
    # Thumb: extended if its tip is farther from the pinky base than its joint is.
    # This works for either hand and any rotation.
    def dist(a, b):
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    thumb = dist(4, 17) > dist(3, 17)
    others = [lm[t].y < lm[p].y for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18])]
    return (thumb, *others)

GESTURES = {
    (False, False, False, False, False): 0, # fist
    (False, True,  False, False, False): 1, # point
    (False, True,  True,  False, False): 2, # peace
    (False,  False, True, False, False): 3, # thumbs up
    (False, True,  False, False, True):  4, # rock
    (True,  True,  True,  True,  True):  5, # open hand
}