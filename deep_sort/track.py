# vim: expandtab:ts=4:sw=4

import numpy as np
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


class TrackOcc:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the heatmap, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,track_cls,
                 feature=None,occ_size=0.4,occ_range=[-40,-40,40,40]):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.track_cls=track_cls

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.occ_size=occ_size
        self.occ_range=np.array(occ_range)
        self.pred=mean
        self.before_align=mean

    # def to_tlwh(self):
    #     """Get current position in bounding box format `(top left x, top left y,
    #     width, height)`.

    #     Returns
    #     -------
    #     ndarray
    #         The bounding box.

    #     """
    #     ret = self.mean[:4].copy()
    #     ret[2] *= ret[3]
    #     ret[:2] -= ret[2:] / 2
    #     return ret

    def to_xy(self):
        """Get current position `(center x, center y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:2].copy()
        return ret
    
    def to_xycls(self):
        """Get current position `(center x, center y)` and cls.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret_xy = np.round(self.mean[:2]).astype(np.int16).tolist()
        return ret_xy+[int(self.track_cls)]
    

    # def to_tlbr(self):
    #     """Get current position in bounding box format `(min x, miny, max x,
    #     max y)`.

    #     Returns
    #     -------
    #     ndarray
    #         The bounding box.

    #     """
    #     ret = self.to_tlwh()
    #     ret[2:] = ret[:2] + ret[2:]
    #     return ret
    def align(self,prev_to_cur_rt):
        """Align the current track with the previous track using the given
        rotation matrix.

        Parameters
        ----------
        prev_to_cur_rt : ndarray
            The rotation matrix to align the current track with the previous
            track.

        """
        self.before_align=self.mean[:2].copy()
        rot=prev_to_cur_rt[:2,:2]
        trans=prev_to_cur_rt[:2,2]

        ego_xy_prev=self.mean[:2]*self.occ_size+self.occ_range[:2]
        ego_xy_cur = np.dot(rot, ego_xy_prev)+trans
        self.mean[:2] = (ego_xy_cur-self.occ_range[:2])/self.occ_size

        # self.covariance[:2, :2] = np.dot(rot, np.dot(self.covariance[:2, :2], rot.T))

    def predict(self, kf,dt):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        state variable:cx,cy,vx,vy
        results in current time ego coordinate
        """

        self.mean, self.covariance = kf.predict(self.mean, self.covariance,dt)
        self.age += 1
        self.time_since_update += 1
        self.pred=self.mean[:2].copy()
        

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xywithvel())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    

class TrackOccEgo:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the heatmap, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,track_cls,
                 feature=None,occ_size=0.4,occ_range=[-40,-40,40,40]):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.track_cls=track_cls

        self.occ_size=occ_size
        self.occ_range=np.array(occ_range)

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.pred=mean
        self.before_align=mean

    # def to_tlwh(self):
    #     """Get current position in bounding box format `(top left x, top left y,
    #     width, height)`.

    #     Returns
    #     -------
    #     ndarray
    #         The bounding box.

    #     """
    #     ret = self.mean[:4].copy()
    #     ret[2] *= ret[3]
    #     ret[:2] -= ret[2:] / 2
    #     return ret

    def to_xy(self):
        """Get current position `(center x, center y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:2].copy()
        return ret
    
    def to_xycls(self):
        """Get current position `(center x, center y)` and cls.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret_xy=np.round((self.mean[:2]-self.occ_range[:2])/self.occ_size).tolist()
        return ret_xy+[int(self.track_cls)]
    

    # def to_tlbr(self):
    #     """Get current position in bounding box format `(min x, miny, max x,
    #     max y)`.

    #     Returns
    #     -------
    #     ndarray
    #         The bounding box.

    #     """
    #     ret = self.to_tlwh()
    #     ret[2:] = ret[:2] + ret[2:]
    #     return ret
    def align(self,prev_to_cur_rt):
        """Align the current track with the previous track using the given
        rotation matrix.

        Parameters
        ----------
        prev_to_cur_rt : ndarray
            The rotation matrix to align the current track with the previous
            track.

        """
        self.before_align=self.mean[:2].copy()
        rot=prev_to_cur_rt[:2,:2]
        trans=prev_to_cur_rt[:2,2]

        ego_xy_prev=self.mean[:2]
        ego_xy_cur = np.dot(rot, ego_xy_prev)+trans
        self.mean[:2] = ego_xy_cur

        # self.covariance[:2, :2] = np.dot(rot, np.dot(self.covariance[:2, :2], rot.T))

    def predict(self, kf,dt):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        state variable:cx,cy,vx,vy
        results in current time ego coordinate
        """

        self.mean, self.covariance = kf.predict(self.mean, self.covariance,dt)
        self.age += 1
        self.time_since_update += 1
        self.pred=self.mean[:2].copy()
        

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xywithvel())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted