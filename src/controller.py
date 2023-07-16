import numpy as np

class Controller:

    def __init__(self, socket):
        self.socket = socket

    def emitControl(self, val):
        if (val == np.array([1, 0, 0])).all():
            self.socket.emit("NXTControl", "right")

        if (val == np.array([0, 1, 0])).all():
            self.socket.emit("NXTControl", "up")

        if (val == np.array([0, 0, 1])).all():
            self.socket.emit("NXTControl", "left")