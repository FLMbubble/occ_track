# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track,TrackOcc


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]


        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1


class TrackerOcc:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric_cos,metric_geo, max_iou_distance=0.7, max_center_distance=10,max_age=30, n_init=3,occ_size=0.4,occ_range=[-40,-40,40,40]):
        self.metric_cos = metric_cos
        self.metric_geo=metric_geo
        self.max_iou_distance = max_iou_distance
        self.max_center_distance=max_center_distance
        self.max_age = max_age
        self.n_init = n_init
        # self.kf = kalman_filter.KalmanFilter()
        self.kf = kalman_filter.KalmanFilterOcc(occ_size=occ_size)
        self.tracks = []
        self._next_id = 1
        self.occ_size=occ_size
        self.occ_range=occ_range

    def align(self,prev_to_cur_rt):
        """Align the current frame with the previous frame.

        This function should be called once every time step, before `update`.
        """
        # Align the current frame with the previous frame
        for track in self.tracks:
            track.align(prev_to_cur_rt)

    def predict(self,dt):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # 延伸现有轨迹
        for track in self.tracks:
            track.predict(self.kf,dt)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        match_track_det=[]
        print("total matches:{}".format(len(matches)))
        # Update track set.
        for track_idx, detection_idx in matches:#用detection结果更新匹配到的轨迹
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            match_track_det.append([self.tracks[track_idx],detections[detection_idx],detection_idx])
        for track_idx in unmatched_tracks:#标记未匹配轨迹
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:#detection对应新轨迹
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

        self.metric_cos.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
    
        return match_track_det

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = self.metric_cos.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix_occ(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        
        def rule_metric(tracks, dets, track_indices, detection_indices):
            det_cls = np.array([dets[i].cls for i in detection_indices]).astype(np.int8)
            track_cls = np.array([tracks[i].track_cls for i in track_indices]).astype(np.int8)
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = (track_cls[:,np.newaxis]==det_cls).astype(np.float16)
            cost_matrix =1-cost_matrix

            return cost_matrix
        
        def distance_metric(tracks, dets, track_indices, detection_indices):
            det_xy= np.stack([dets[i].xy for i in detection_indices])
            track_xy = np.stack([tracks[i].mean[:2] for i in track_indices])
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = track_xy[:,np.newaxis,:]-det_xy[np.newaxis,:]
            cost_matrix=np.sqrt(np.sum(cost_matrix**2,axis=-1))

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade(
        #         gated_metric, self.metric.matching_threshold, self.max_age,
        #         self.tracks, detections, confirmed_tracks)
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade_with_label(
        #         gated_metric, rule_metric,self.metric_cos.matching_threshold, self.max_age,
        #         self.tracks, detections, confirmed_tracks)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade_with_label(
                distance_metric, rule_metric,self.max_center_distance, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using center distance.

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_multi_cost_matching(
                distance_metric,rule_metric, self.max_center_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # mean, covariance = self.kf.initiate(detection.to_xyah())
        mean, covariance = self.kf.initiate(detection.to_xywithvel())#TO DO
        # self.tracks.append(Track(
        #     mean, covariance, self._next_id, self.n_init, self.max_age,
        #     detection.feature))
        self.tracks.append(TrackOcc(
            mean, covariance, self._next_id, self.n_init, self.max_age,detection.cls,
            detection.feature,occ_size=self.occ_size,occ_range=self.occ_range))
        self._next_id += 1



class TrackerOccEgo:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric_cos,metric_geo, max_iou_distance=0.7, max_center_distance=10,max_age=30, n_init=2,occ_size=0.4,occ_range=[-40,-40,40,40]):
        self.metric_cos = metric_cos
        self.metric_geo=metric_geo
        self.max_iou_distance = max_iou_distance
        self.max_center_distance=max_center_distance
        self.max_age = max_age
        self.n_init = n_init

        # self.kf = kalman_filter.KalmanFilter()
        self.kf = kalman_filter.KalmanFilterOccEgo()
        self.tracks = []
        self._next_id = 1
        self.occ_size=occ_size
        self.occ_range=occ_range

    def align(self,prev_to_cur_rt):
        """Align the current frame with the previous frame.

        This function should be called once every time step, before `update`.
        """
        # Align the current frame with the previous frame
        for track in self.tracks:
            track.align(prev_to_cur_rt)

    def predict(self,dt):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # 延伸现有轨迹
        for track in self.tracks:
            track.predict(self.kf,dt)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        match_track_det=[]
        print("total matches:{}".format(len(matches)))
        # Update track set.
        for track_idx, detection_idx in matches:#用detection结果更新匹配到的轨迹
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            match_track_det.append([self.tracks[track_idx],detections[detection_idx],detection_idx])
        for track_idx in unmatched_tracks:#标记未匹配轨迹
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:#detection对应新轨迹
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

        self.metric_cos.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
    
        return match_track_det

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = self.metric_cos.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix_occ(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        
        def rule_metric(tracks, dets, track_indices, detection_indices):
            det_cls = np.array([dets[i].cls for i in detection_indices]).astype(np.int8)
            track_cls = np.array([tracks[i].track_cls for i in track_indices]).astype(np.int8)
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = (track_cls[:,np.newaxis]==det_cls).astype(np.float16)
            cost_matrix =1-cost_matrix

            return cost_matrix
        
        def distance_metric(tracks, dets, track_indices, detection_indices):
            det_xy= np.stack([dets[i].xy for i in detection_indices])
            track_xy = np.stack([tracks[i].mean[:2] for i in track_indices])
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = track_xy[:,np.newaxis,:]-det_xy[np.newaxis,:]
            cost_matrix=np.sqrt(np.sum(cost_matrix**2,axis=-1))

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade(
        #         gated_metric, self.metric.matching_threshold, self.max_age,
        #         self.tracks, detections, confirmed_tracks)
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade_with_label(
        #         gated_metric, rule_metric,self.metric_cos.matching_threshold, self.max_age,
        #         self.tracks, detections, confirmed_tracks)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade_with_label(
                distance_metric, rule_metric,self.max_center_distance, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using center distance.

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_multi_cost_matching(
                distance_metric,rule_metric, self.max_center_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # mean, covariance = self.kf.initiate(detection.to_xyah())
        mean, covariance = self.kf.initiate(detection.to_xywithvel())#TO DO
        # self.tracks.append(Track(
        #     mean, covariance, self._next_id, self.n_init, self.max_age,
        #     detection.feature))
        self.tracks.append(TrackOcc(
            mean, covariance, self._next_id, self.n_init, self.max_age,detection.cls,
            detection.feature,occ_size=self.occ_size,occ_range=self.occ_range))
        self._next_id += 1