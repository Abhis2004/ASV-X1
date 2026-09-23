import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import scipy.linalg


class KalmanFilter:
    """
    Kalman Filter for bounding box tracking in image space.
    State vector: [cx, cy, a, h, vx, vy, va, vh]
    Where (cx, cy) is box center, 'a' is aspect ratio (w/h), 'h' is height.
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviations for position and velocity scaling
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict state for next time step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step using measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(self._update_mat, covariance), check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class CameraMotionCompensation:
    """
    Global Motion Compensation (GMC) for aerial drone movement.
    Uses sparse optical flow to compute affine warp matrix between frames.
    """

    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.prev_frame = None

    def apply(self, frame, tracks):
        """
        Estimate camera warp matrix and adjust track states (Kalman mean & covariance).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self.prev_frame is None or self.prev_frame.shape != gray.shape:
            self.prev_frame = gray
            return

        # Find features on previous frame
        mask = np.ones_like(self.prev_frame, dtype=np.uint8) * 255
        # Mask out current track bounding boxes to focus on static background
        for track in tracks:
            if track.is_confirmed() or track.is_tentative():
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(frame.shape[1], x2), min(frame.shape[0], y2)), 0, -1)

        kp1 = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10, mask=mask)

        if kp1 is None or len(kp1) < 10:
            self.prev_frame = gray
            return

        kp2, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, kp1, None)

        if kp2 is None or status is None:
            self.prev_frame = gray
            return

        good_prev = kp1[status == 1]
        good_curr = kp2[status == 1]

        if len(good_prev) < 8:
            self.prev_frame = gray
            return

        # Compute partial affine transformation (translation, rotation, scale)
        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M is not None:
            R = M[:2, :2]
            t = M[:2, 2]

            # Warp tracks according to camera movement
            for track in tracks:
                # Warp Kalman mean (cx, cy)
                cx, cy = track.mean[0], track.mean[1]
                warped_pos = np.dot(R, np.array([cx, cy])) + t
                track.mean[0] = warped_pos[0]
                track.mean[1] = warped_pos[1]

                # Rotate Kalman velocities (vx, vy)
                vx, vy = track.mean[4], track.mean[5]
                warped_vel = np.dot(R, np.array([vx, vy]))
                track.mean[4] = warped_vel[0]
                track.mean[5] = warped_vel[1]

                # Update trajectory memory
                if len(track.history) > 0:
                    hist_arr = np.array(track.history)
                    warped_hist = np.dot(hist_arr, R.T) + t
                    track.history = [tuple(p) for p in warped_hist]

        self.prev_frame = gray


def calculate_diou_matrix(boxes1, boxes2):
    """
    Calculates Distance-IoU (DIoU) cost matrix between two sets of boxes (x1, y1, x2, y2).
    Returns cost matrix of shape (N, M) where Cost = 1 - DIoU.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)))

    b1 = np.ascontiguousarray(boxes1, dtype=np.float32)
    b2 = np.ascontiguousarray(boxes2, dtype=np.float32)

    # Box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0:1], b1[:, 1:2], b1[:, 2:3], b1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0:1].T, b2[:, 1:2].T, b2[:, 2:3].T, b2[:, 3:4].T

    # Intersection
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # Centers
    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0

    center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Enclosing box
    enc_x1 = np.minimum(b1_x1, b2_x1)
    enc_y1 = np.minimum(b1_y1, b2_y1)
    enc_x2 = np.maximum(b1_x2, b2_x2)
    enc_y2 = np.maximum(b1_y2, b2_y2)

    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    diou = iou - (center_dist_sq / enc_diag_sq)
    cost = 1.0 - diou
    return cost


class TrackState:
    Tentative = 1
    Confirmed = 2
    Lost = 3
    Deleted = 4


class Track:
    """
    Track instance managing individual target lifecycle and trajectory.
    """
    _id_counter = 1

    def __init__(self, mean, covariance, cls_id, conf, max_history=30):
        self.track_id = Track._id_counter
        Track._id_counter += 1

        self.mean = mean
        self.covariance = covariance
        self.cls = int(cls_id)
        self.conf = float(conf)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        self.history = []
        self.max_history = max_history

        # Record center
        cx, cy = mean[0], mean[1]
        self.history.append((cx, cy))

    def to_tlbr(self):
        """Convert state [cx, cy, a, h] to (top-left, bottom-right) [x1, y1, x2, y2]."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self, kf):
        """Predict next Kalman state."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        cx, cy = self.mean[0], self.mean[1]
        self.history.append((cx, cy))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def update(self, kf, detection):
        """Update Kalman state with matched detection [x1, y1, x2, y2, conf, cls]."""
        x1, y1, x2, y2, conf, cls_id = detection
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        a = w / h

        measurement = np.array([cx, cy, a, h])
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0
        self.conf = float(conf)
        self.cls = int(cls_id)

        if self.state == TrackState.Tentative and self.hits >= 2:
            self.state = TrackState.Confirmed

    def mark_missed(self, max_age):
        """Handle unmatched frame."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > max_age:
            self.state = TrackState.Deleted
        else:
            self.state = TrackState.Lost

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_lost(self):
        return self.state == TrackState.Lost


class DroneTracker:
    """
    Professional Drone Multi-Object Tracker optimized for VisDrone aerial datasets.
    Combines:
      - Camera Motion Compensation (GMC) via sparse LK optical flow
      - Scaled-variance Kalman Filtering
      - ByteTrack-style two-stage confidence association
      - DIoU (Distance IoU) matching cost
    """

    def __init__(
        self,
        det_thresh=0.25,
        low_thresh=0.08,
        match_thresh=0.75,
        max_age=30,
        min_hits=2,
        use_gmc=True,
    ):
        self.det_thresh = det_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_gmc = use_gmc

        self.kf = KalmanFilter()
        self.gmc = CameraMotionCompensation() if use_gmc else None
        self.tracks = []

    def update(self, detections, frame=None):
        """
        Update tracker with new frame detections.
        detections: np.ndarray or list of [x1, y1, x2, y2, conf, cls]
        frame: BGR numpy image frame (optional, required if GMC is enabled)
        Returns: list of active confirmed/lost Track objects
        """
        if len(detections) == 0:
            detections = np.empty((0, 6))
        else:
            detections = np.array(detections, dtype=np.float32)

        # 1. Camera Motion Compensation
        if self.gmc is not None and frame is not None:
            self.gmc.apply(frame, self.tracks)

        # 2. Predict all track states via Kalman Filter
        for track in self.tracks:
            track.predict(self.kf)

        # 3. Split detections into high confidence & low confidence
        high_mask = detections[:, 4] >= self.det_thresh
        low_mask = (detections[:, 4] >= self.low_thresh) & (~high_mask)

        dets_high = detections[high_mask]
        dets_low = detections[low_mask]

        # Separate tracks into confirmed and unconfirmed/lost
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if t.is_tentative() or t.is_lost()]

        # ----------------------------------------------------
        # First Association: Confirmed tracks <-> High-conf detections
        # ----------------------------------------------------
        track_boxes = [t.to_tlbr() for t in confirmed_tracks]
        det_high_boxes = [d[:4] for d in dets_high]

        cost_matrix = calculate_diou_matrix(track_boxes, det_high_boxes)

        matches_a, unassigned_tracks_a, unassigned_dets_high = self._linear_assignment(
            cost_matrix, self.match_thresh, confirmed_tracks, dets_high
        )

        for trk_idx, det_idx in matches_a:
            confirmed_tracks[trk_idx].update(self.kf, dets_high[det_idx])

        # ----------------------------------------------------
        # Second Association: Remaining confirmed tracks <-> Low-conf detections
        # ----------------------------------------------------
        remain_confirmed = [confirmed_tracks[i] for i in unassigned_tracks_a]
        det_low_boxes = [d[:4] for d in dets_low]
        track_boxes_remain = [t.to_tlbr() for t in remain_confirmed]

        cost_matrix_low = calculate_diou_matrix(track_boxes_remain, det_low_boxes)

        matches_b, unassigned_tracks_b, _ = self._linear_assignment(
            cost_matrix_low, 0.85, remain_confirmed, dets_low
        )

        for trk_idx, det_idx in matches_b:
            remain_confirmed[trk_idx].update(self.kf, dets_low[det_idx])

        # Mark confirmed tracks that missed both associations
        for trk_idx in unassigned_tracks_b:
            remain_confirmed[trk_idx].mark_missed(self.max_age)

        # ----------------------------------------------------
        # Third Association: Unassigned high-conf detections <-> Unconfirmed/Lost tracks
        # ----------------------------------------------------
        remain_dets_high = [dets_high[i] for i in unassigned_dets_high]
        remain_dets_high_boxes = [d[:4] for d in remain_dets_high]
        unconfirmed_boxes = [t.to_tlbr() for t in unconfirmed_tracks]

        cost_matrix_unconf = calculate_diou_matrix(unconfirmed_boxes, remain_dets_high_boxes)

        matches_c, unassigned_tracks_c, unassigned_dets_c = self._linear_assignment(
            cost_matrix_unconf, 0.70, unconfirmed_tracks, remain_dets_high
        )

        for trk_idx, det_idx in matches_c:
            unconfirmed_tracks[trk_idx].update(self.kf, remain_dets_high[det_idx])

        for trk_idx in unassigned_tracks_c:
            unconfirmed_tracks[trk_idx].mark_missed(self.max_age)

        # ----------------------------------------------------
        # Create new tracks for remaining unmatched high-conf detections
        # ----------------------------------------------------
        for det_idx in unassigned_dets_c:
            det = remain_dets_high[det_idx]
            x1, y1, x2, y2, conf, cls_id = det
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            a = w / h

            measurement = np.array([cx, cy, a, h])
            mean, covariance = self.kf.initiate(measurement)
            new_track = Track(mean, covariance, cls_id, conf)
            if self.min_hits <= 1:
                new_track.state = TrackState.Confirmed
            self.tracks.append(new_track)

        # Clean up deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted]

        # Return active tracks (confirmed or recently lost)
        return [t for t in self.tracks if t.is_confirmed() or t.is_lost()]

    def _linear_assignment(self, cost_matrix, thresh, tracks, detections):
        """
        Solves optimal bipartite matching using Hungarian algorithm.
        Rejects matches exceeding cost threshold.
        """
        if cost_matrix.size == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.nan_to_num(cost_matrix, nan=1e5)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unassigned_tracks = set(range(len(tracks)))
        unassigned_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                unassigned_tracks.discard(r)
                unassigned_dets.discard(c)

        return matches, list(unassigned_tracks), list(unassigned_dets)
