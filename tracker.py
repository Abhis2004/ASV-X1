import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import scipy.linalg
from collections import deque


class AppearanceExtractor:
    """Lightweight spatial HSV histogram extractor for object re-identification."""
    def __init__(self, crop_size=(64, 64), h_bins=16, s_bins=8, grid=2):
        self.crop_size = crop_size
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.grid = grid
        self.feature_dim = grid * grid * (h_bins + s_bins)

    def extract(self, frame, bbox):
        if frame is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
            
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return np.zeros(self.feature_dim, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
            
        crop = cv2.resize(crop, self.crop_size)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h, w = self.crop_size
        cell_h, cell_w = h // self.grid, w // self.grid

        features = []
        for i in range(self.grid):
            for j in range(self.grid):
                cell = hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist_h = cv2.calcHist([cell], [0], None, [self.h_bins], [0, 180])
                hist_s = cv2.calcHist([cell], [1], None, [self.s_bins], [0, 256])
                cv2.normalize(hist_h, hist_h)
                cv2.normalize(hist_s, hist_s)
                features.append(hist_h.flatten())
                features.append(hist_s.flatten())

        feature_vec = np.concatenate(features)
        norm = np.linalg.norm(feature_vec)
        if norm > 0:
            feature_vec = feature_vec / norm
        return feature_vec


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
                    track.history = deque([tuple(p) for p in warped_hist], maxlen=track.max_history)

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


def calculate_appearance_cost(tracks, det_features, det_boxes):
    """
    Calculates cost matrix using Cosine similarity of Appearance features and direction gating.
    """
    if len(tracks) == 0 or len(det_features) == 0:
        return np.empty((len(tracks), len(det_features)))
    
    cost_matrix = np.zeros((len(tracks), len(det_features)), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, (det_feat, det_box) in enumerate(zip(det_features, det_boxes)):
            if track.appearance is None or det_feat is None or np.sum(det_feat) == 0:
                cost_matrix[i, j] = 1.0
                continue
            
            # 1. Cosine distance
            cos_sim = np.dot(track.appearance, det_feat)
            app_cost = 1.0 - cos_sim
            
            # 2. Direction-Aware Gating penalty
            last_cx, last_cy = track.last_confirmed_position
            vx, vy = track.last_confirmed_velocity
            
            det_cx = (det_box[0] + det_box[2]) / 2.0
            det_cy = (det_box[1] + det_box[3]) / 2.0
            
            dx = det_cx - last_cx
            dy = det_cy - last_cy
            
            # If track had significant velocity, check if det is in same direction
            v_norm = np.hypot(vx, vy)
            d_norm = np.hypot(dx, dy)
            
            if v_norm > 2.0 and d_norm > 10.0:
                cos_dir = (vx*dx + vy*dy) / (v_norm * d_norm + 1e-6)
                if cos_dir < 0: # Moving opposite direction
                    app_cost += 0.3 * abs(cos_dir)
                    
            cost_matrix[i, j] = app_cost
            
    return cost_matrix


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

    def __init__(self, mean, covariance, cls_id, conf, max_history=30, feature=None):
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

        self.max_history = max_history
        self.history = deque(maxlen=max_history)

        # Stage 4 Re-ID Extensions
        self.appearance = feature
        self.confirmed_frames = 0
        self.last_confirmed_position = (mean[0], mean[1])
        self.last_confirmed_velocity = (0.0, 0.0)

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

    def update_appearance(self, new_feature, momentum=0.85):
        """EMA update for appearance robust to short-term variations."""
        if new_feature is None or np.sum(new_feature) == 0:
            return
        if self.appearance is None:
            self.appearance = new_feature
        else:
            self.appearance = momentum * self.appearance + (1.0 - momentum) * new_feature
            norm = np.linalg.norm(self.appearance)
            if norm > 0:
                self.appearance = self.appearance / norm

    def update(self, kf, detection, feature=None):
        """Update Kalman state with matched detection [x1, y1, x2, y2, conf, cls]."""
        x1, y1, x2, y2, conf, cls_id = detection
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        a = w / h

        # Stage 4 Update Extensions: Compute velocity & log last confirmed pos
        vx = cx - self.last_confirmed_position[0]
        vy = cy - self.last_confirmed_position[1]
        self.last_confirmed_velocity = (vx, vy)
        self.last_confirmed_position = (cx, cy)
        self.confirmed_frames += 1

        measurement = np.array([cx, cy, a, h])
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0
        self.conf = float(conf)
        self.cls = int(cls_id)
        
        if feature is not None:
            self.update_appearance(feature)

        if self.state == TrackState.Tentative and self.hits >= 2:
            self.state = TrackState.Confirmed

    def mark_missed(self, base_max_age):
        """Handle unmatched frame with adaptive track lifetime."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            return
            
        # Adaptive max age logic
        adaptive_max_age = base_max_age
        if self.confirmed_frames > 15 and self.appearance is not None:
            adaptive_max_age = int(base_max_age * 2.5)
        elif self.confirmed_frames > 5:
            adaptive_max_age = int(base_max_age * 1.5)
            
        if self.time_since_update > adaptive_max_age:
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
      - Stage 4 Appearance-Based Re-ID (Handles long-term bridge/tunnel occlusions)
    """

    def __init__(
        self,
        det_thresh=0.25,
        low_thresh=0.08,
        match_thresh=0.75,
        max_age=30,
        min_hits=2,
        use_gmc=True,
        reid_thresh=0.45,
        long_lost_frames=8
    ):
        self.det_thresh = det_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_gmc = use_gmc
        self.reid_thresh = reid_thresh
        self.long_lost_frames = long_lost_frames

        self.kf = KalmanFilter()
        self.gmc = CameraMotionCompensation() if use_gmc else None
        self.app_extractor = AppearanceExtractor()
        self.tracks = []

    def update(self, detections, frame=None):
        """
        Update tracker with new frame detections.
        detections: np.ndarray or list of [x1, y1, x2, y2, conf, cls]
        frame: BGR numpy image frame (optional, required if GMC is enabled or Appearance Re-ID)
        Returns: list of active confirmed/lost Track objects
        """
        if len(detections) == 0:
            detections = np.empty((0, 6))
        else:
            detections = np.array(detections, dtype=np.float32)
            
        # 1. Extract appearance features for all detections
        det_features = []
        if frame is not None and len(detections) > 0:
            for det in detections:
                feat = self.app_extractor.extract(frame, det[:4])
                det_features.append(feat)
        else:
            det_features = [None] * len(detections)

        # 2. Camera Motion Compensation
        if self.gmc is not None and frame is not None:
            self.gmc.apply(frame, self.tracks)

        # 3. Predict all track states via Kalman Filter
        for track in self.tracks:
            track.predict(self.kf)

        # 4. Split detections into high confidence & low confidence
        high_mask = detections[:, 4] >= self.det_thresh
        low_mask = (detections[:, 4] >= self.low_thresh) & (~high_mask)

        dets_high = detections[high_mask]
        dets_low = detections[low_mask]
        
        feats_high = [det_features[i] for i in range(len(detections)) if high_mask[i]]
        feats_low = [det_features[i] for i in range(len(detections)) if low_mask[i]]

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
            confirmed_tracks[trk_idx].update(self.kf, dets_high[det_idx], feats_high[det_idx])

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
            remain_confirmed[trk_idx].update(self.kf, dets_low[det_idx], feats_low[det_idx])

        # Mark confirmed tracks that missed both associations
        for trk_idx in unassigned_tracks_b:
            remain_confirmed[trk_idx].mark_missed(self.max_age)

        # ----------------------------------------------------
        # Third Association: Unassigned high-conf detections <-> Recently Lost / Tentative tracks
        # ----------------------------------------------------
        remain_dets_high = [dets_high[i] for i in unassigned_dets_high]
        remain_feats_high = [feats_high[i] for i in unassigned_dets_high]
        remain_dets_high_boxes = [d[:4] for d in remain_dets_high]
        
        # Filter unconfirmed tracks: Stage 3 handles recent occlusion or tentative
        stage3_tracks = [t for t in unconfirmed_tracks if t.is_tentative() or (t.is_lost() and t.time_since_update < self.long_lost_frames)]
        # Filter unconfirmed tracks: Stage 4 handles long-term occlusion
        stage4_tracks = [t for t in unconfirmed_tracks if t.is_lost() and t.time_since_update >= self.long_lost_frames and t.appearance is not None]
        other_unconfirmed = [t for t in unconfirmed_tracks if t not in stage3_tracks and t not in stage4_tracks]

        stage3_boxes = [t.to_tlbr() for t in stage3_tracks]
        cost_matrix_unconf = calculate_diou_matrix(stage3_boxes, remain_dets_high_boxes)

        matches_c, unassigned_tracks_c, unassigned_dets_c = self._linear_assignment(
            cost_matrix_unconf, 0.70, stage3_tracks, remain_dets_high
        )

        for trk_idx, det_idx in matches_c:
            stage3_tracks[trk_idx].update(self.kf, remain_dets_high[det_idx], remain_feats_high[det_idx])

        for trk_idx in unassigned_tracks_c:
            stage3_tracks[trk_idx].mark_missed(self.max_age)
            
        # ----------------------------------------------------
        # Fourth Association (NEW): Long-lost tracks <-> Remaining high-conf detections (Appearance Re-ID)
        # ----------------------------------------------------
        remain_dets_stage4 = [remain_dets_high[i] for i in unassigned_dets_c]
        remain_feats_stage4 = [remain_feats_high[i] for i in unassigned_dets_c]
        remain_dets_stage4_boxes = [d[:4] for d in remain_dets_stage4]
        
        if len(stage4_tracks) > 0 and len(remain_dets_stage4) > 0:
            cost_matrix_reid = calculate_appearance_cost(stage4_tracks, remain_feats_stage4, remain_dets_stage4_boxes)
            
            # Additional Class Gating for Stage 4
            for i, track in enumerate(stage4_tracks):
                for j, det in enumerate(remain_dets_stage4):
                    if track.cls != int(det[5]):  # Class mismatch prevents re-ID match
                        cost_matrix_reid[i, j] = 1.0
                        
            matches_d, unassigned_tracks_d, unassigned_dets_d = self._linear_assignment(
                cost_matrix_reid, self.reid_thresh, stage4_tracks, remain_dets_stage4
            )
            
            for trk_idx, det_idx in matches_d:
                # Re-confirm the long lost track successfully via Re-ID!
                stage4_tracks[trk_idx].update(self.kf, remain_dets_stage4[det_idx], remain_feats_stage4[det_idx])
                stage4_tracks[trk_idx].state = TrackState.Confirmed
                
            for trk_idx in unassigned_tracks_d:
                stage4_tracks[trk_idx].mark_missed(self.max_age)
                
            unassigned_dets_final = [unassigned_dets_c[i] for i in unassigned_dets_d]
        else:
            for trk in stage4_tracks:
                trk.mark_missed(self.max_age)
            unassigned_dets_final = unassigned_dets_c
            
        # Ensure tracks that don't fit stage 3 or 4 get processed (should just hit max_age)
        for trk in other_unconfirmed:
            trk.mark_missed(self.max_age)

        # ----------------------------------------------------
        # Create new tracks for remaining unmatched high-conf detections
        # ----------------------------------------------------
        for det_idx in unassigned_dets_final:
            det = remain_dets_high[det_idx]
            feat = remain_feats_high[det_idx]
            x1, y1, x2, y2, conf, cls_id = det
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            a = w / h

            measurement = np.array([cx, cy, a, h])
            mean, covariance = self.kf.initiate(measurement)
            new_track = Track(mean, covariance, cls_id, conf, feature=feat)
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
