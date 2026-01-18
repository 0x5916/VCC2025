"""
claim_downer.py

Contains the Claimdowner class that manages logic for "found" and "down" states.
"""

class Claimdowner:
    """
    Example class that tracks a 'found' state and a 'down' action.
    Modify as needed for your detection logic.
    """

    def __init__(self):
        self._count = {}
        self._state = False  # Indicates whether something was recently "found"

    def found(self, label):
        """
        Called when a marble is 'found'. Sets an internal state.
        """
        self._state = True
        if not label in self._count:
            self._count[label] = 0
        self._count[label] += 1

    def total_count(self):
        return sum(self._count.values())

    def count(self, label):
        if label in self._count:
            return self._count[label]
        return 0
    
    def max_count_label(self):
        return max(self._count, key=self._count.get)

    def reset(self) -> bool:
        """
        Called when a marble transitions to the 'down' position.
        Returns True if the internal state was set, then resets it.
        """
        if self._state:
            self._state = False
            self._count = {}
            return True
        
        return False