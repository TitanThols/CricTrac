import cv2
import numpy as np

class PatchOpticalFlow:
    def __init__(self, max_corners=150, quality=0.01, min_dist=7, min_points=5):
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_pts = None
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.min_points = min_points
        self.frame_count = 0
        self.refresh_interval = 10  # Refresh points every N frames

    def init(self, gray, rect, expand=1.2):
        """Initialize or reinitialize optical flow tracking patch"""
        (cx, cy), (w, h), angle = rect
        size = int(max(w, h) * expand)
        x1 = int(max(0, cx - size/2))
        y1 = int(max(0, cy - size/2))
        x2 = int(min(gray.shape[1], cx + size/2))
        y2 = int(min(gray.shape[0], cy + size/2))
        
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            self.prev_pts = None
            return False
            
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=self.max_corners,
                                      qualityLevel=self.quality, minDistance=self.min_dist)
        if pts is None or len(pts) < self.min_points:
            self.prev_pts = None
            return False
            
        pts[:,0,0] += x1
        pts[:,0,1] += y1
        self.prev_pts = pts
        self.prev_gray = gray.copy()
        self.frame_count = 0
        return True

    def step(self, gray, current_rect=None):
        """Track points and return displacement. Optionally refresh points if rect provided."""
        if self.prev_pts is None or self.prev_gray is None:
            return None
        
        # Track points
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        if nextPts is None:
            self.prev_pts = None
            return None
            
        status = status.reshape(-1)
        good_prev = self.prev_pts[status==1]
        good_next = nextPts[status==1]
        
        # If too few points, try to refresh if we have current position
        if len(good_next) < self.min_points:
            if current_rect is not None:
                # Try to reinitialize
                if self.init(gray, current_rect):
                    return (0.0, 0.0, self.prev_pts)  # No displacement this frame
            self.prev_pts = None
            return None
        
        # Use RANSAC-based estimation if enough points, otherwise median
        if len(good_next) >= 10:
            # Estimate affine transform with RANSAC
            M, inliers = cv2.estimateAffinePartial2D(
                good_prev.reshape(-1, 2), 
                good_next.reshape(-1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            if M is not None and inliers is not None:
                inlier_mask = inliers.reshape(-1)
                inlier_prev = good_prev[inlier_mask == 1]
                inlier_next = good_next[inlier_mask == 1]
                if len(inlier_next) >= self.min_points:
                    dx = np.median(inlier_next[:,0,0] - inlier_prev[:,0,0])
                    dy = np.median(inlier_next[:,0,1] - inlier_prev[:,0,1])
                    # Keep only inliers for next frame
                    self.prev_pts = inlier_next.reshape(-1, 1, 2)
                else:
                    dx = np.median(good_next[:,0,0] - good_prev[:,0,0])
                    dy = np.median(good_next[:,0,1] - good_prev[:,0,1])
                    self.prev_pts = good_next.reshape(-1, 1, 2)
            else:
                dx = np.median(good_next[:,0,0] - good_prev[:,0,0])
                dy = np.median(good_next[:,0,1] - good_prev[:,0,1])
                self.prev_pts = good_next.reshape(-1, 1, 2)
        else:
            dx = np.median(good_next[:,0,0] - good_prev[:,0,0])
            dy = np.median(good_next[:,0,1] - good_prev[:,0,1])
            self.prev_pts = good_next.reshape(-1, 1, 2)
        
        self.prev_gray = gray.copy()
        self.frame_count += 1
        
        # Periodic refresh if we have current rect
        if current_rect is not None and self.frame_count >= self.refresh_interval:
            self.init(gray, current_rect)
        
        return dx, dy, self.prev_pts
    
    def reset(self):
        """Clear all tracking state"""
        self.prev_pts = None
        self.prev_gray = None
        self.frame_count = 0