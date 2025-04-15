# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection,DetectionOcc
from deep_sort.tracker import Tracker,TrackerOcc


def sort_files(detection_batch_dir,sort_out_dir='./detections/MY_PANOOCC'):
    detection_all=os.listdir(detection_batch_dir)
    scene_dict=dict()
    time_dict=dict()

    frame_id=0
    frame_id_map=dict()
    if not os.path.exists(sort_out_dir):
        os.makedirs(sort_out_dir)
    for detection_f in detection_all:
        detection_data=np.load(os.path.join(detection_batch_dir,detection_f))
        scene_name=detection_data['scene_name'].item()
        if not scene_name in scene_dict.keys():
            scene_dict[scene_name]=[]
            time_dict[scene_name]=[]
            frame_id_map[scene_name]=frame_id
            frame_id+=1
        
        inst_num=detection_data['inst_xyz'].shape[0]
        single_info=np.zeros((inst_num,72),dtype=np.float32)-1#frame_id,id,cx,cy,cls,conf,vx,vy,feat_64
        time_info=np.zeros((inst_num),dtype=np.float64)+detection_data['timestamp']
        single_info[:,2:4]=detection_data['inst_xyz'][...,:2]
        single_info[:,4]=detection_data['inst_cls']
        single_info[:,5]=detection_data['inst_conf']

        bev_vel=detection_data['vel'].squeeze().transpose(2,1,0)
        single_info[:,6:8]=bev_vel[detection_data['inst_xyz'][:,0],detection_data['inst_xyz'][:,1]]
        
        bev_feat=detection_data['bev_feat'].squeeze().transpose(2,1,0)
        single_info[:,8:]=bev_feat[detection_data['inst_xyz'][:,0],detection_data['inst_xyz'][:,1]]

        scene_dict[scene_name].append(single_info)
        time_dict[scene_name].append(time_info)
    
    for scene_id in scene_dict.keys():
        scene_data=np.concatenate(scene_dict[scene_id],axis=0)
        time_data=np.concatenate(time_dict[scene_id],axis=0)
        idx=np.argsort(time_data)

        new_scene_data=scene_data[idx]
        new_time_data=time_data[idx]

        unique_time=sorted(set(new_time_data))
        idx_mapping={value:idx for idx,value in enumerate(unique_time)}

        new_frame_idx=[idx_mapping[value] for value in new_time_data]
        new_scene_data[:,0]=new_frame_idx


        np.save(os.path.join(sort_out_dir,'det-{}.npy'.format(scene_id)),new_scene_data)
        np.save(os.path.join(sort_out_dir,'time-{}.npy'.format(scene_id)),new_time_data)

    print("sort finished!")

        
        

def gather_sequence_info(detection_file,bev_size):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.-->
        * groundtruth: A numpy array of ground truth in MOTChallenge format.-->
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.-->
        * max_frame_idx: Index of the last frame.-->

    """


    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None




    min_frame_idx=int(detections[:,0].min())
    max_frame_idx=int(detections[:,0].max())

    update_ms=100

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    sequence_dir=detection_file.split('.')[0].split('-')[-1]
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "detections": detections,
        "groundtruth": groundtruth,
        "bev_size": bev_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, timestamp,min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns list:[bbox info,confidence,feature]
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        # bbox, confidence, feature = row[2:6], row[6], row[10:]
        xy, cls,confidence, v_xy,feature = row[2:4], row[4],row[5],row[6:8], row[8:]

        detection_list.append(DetectionOcc(xy, cls,confidence,v_xy,feature,timestamp))
    return detection_list


def run(detection_file, output_file, min_confidence,max_cosine_distance,nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maximum suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(detection_file)
    metric_cos = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    metric_geo=1
    tracker = TrackerOcc(metric_cos,metric_geo)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maximum suppression.
        # boxes = np.array([d.tlwh for d in detections])
        centers=np.array([d.xy for d in detections])
        scores = np.array([d.confidence for d in detections])


        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.TO DO
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # bbox = track.to_tlwh()
            info = track.get_info()
            # results.append([
            #     frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            results.append([
                frame_idx, track.track_id, info[0], info[1], info[2]])#info:cx,cy,cls

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # run(
    #     args.sequence_dir, args.detection_file, args.output_file,
    #     args.min_confidence, args.nms_max_overlap, args.min_detection_height,
    #     args.max_cosine_distance, args.nn_budget, args.display)

    sort_files('./detections/track_data')