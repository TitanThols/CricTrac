import numpy as np
import cv2

class KalmanCentroid:
    def __init__(self, dt=1.0, process_noise=0.5, meas_noise=5.0):
        # 4 state: x, y, vx, vy ; 2 measurements: x, y
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],
                                             [0,1,0,dt],
                                             [0,0,1,0],
                                             [0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x],[y],[0.0],[0.0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        p = self.kf.predict()
        return float(p[0,0]), float(p[1,0])

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        post = self.kf.correct(meas)
        return float(post[0,0]), float(post[1,0])
    
    def reset(self):
        self.initialized = False
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)