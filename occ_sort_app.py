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
    aux_dict=dict()


    if not os.path.exists(sort_out_dir):
        os.makedirs(sort_out_dir)
    for detection_f in detection_all:
        detection_data=np.load(os.path.join(detection_batch_dir,detection_f))
        scene_name=detection_data['scene_name'].item()
        if not scene_name in scene_dict.keys():
            scene_dict[scene_name]=[]
            time_dict[scene_name]=[]
            aux_dict[scene_name]=dict()


            
        
        inst_num=detection_data['inst_xyz'].shape[0]
        single_info=np.zeros((inst_num,72),dtype=np.float32)-1#frame_id,id,cx,cy,cls,conf,vx,vy,feat_64
        time_info=np.zeros((inst_num),dtype=np.float64)+detection_data['timestamp']
 
        single_info[:,1]=detection_data['inst_id']
        single_info[:,2:4]=detection_data['inst_xyz'][...,:2]
        single_info[:,4]=detection_data['inst_cls']
        single_info[:,5]=detection_data['inst_conf']

        bev_vel=detection_data['vel'].squeeze().transpose(2,1,0)
        single_info[:,6:8]=bev_vel[detection_data['inst_xyz'][:,0],detection_data['inst_xyz'][:,1]]
        
        bev_feat=detection_data['bev_feat'].squeeze().transpose(2,1,0)
        single_info[:,8:]=bev_feat[detection_data['inst_xyz'][:,0],detection_data['inst_xyz'][:,1]]

        indices_sort=single_info[:,4].argsort()
        single_info=single_info[indices_sort]

        scene_dict[scene_name].append(single_info)
        time_dict[scene_name].append(time_info)
        aux_dict[scene_name][detection_f]=float(detection_data['timestamp'])
    
    for scene_id in scene_dict.keys():
        scene_data=np.concatenate(scene_dict[scene_id],axis=0)
        time_data=np.concatenate(time_dict[scene_id],axis=0)

        idx=np.argsort(time_data)

        new_scene_data=scene_data[idx]
        new_time_data=time_data[idx]
        aux_data=aux_dict[scene_id]

        unique_time=sorted(set(new_time_data))
        idx_mapping={value:idx for idx,value in enumerate(unique_time)}


        new_frame_idx=[idx_mapping[value] for value in new_time_data]
        new_scene_data[:,0]=new_frame_idx
        new_aux_data=dict()
        for key,val in aux_data.items():
            new_aux_data[str(idx_mapping[aux_data[key]])]=key

        np.save(os.path.join(sort_out_dir,'det-{}.npy'.format(scene_id)),new_scene_data)
        np.save(os.path.join(sort_out_dir,'time-{}.npy'.format(scene_id)),new_time_data)
        np.savez(os.path.join(sort_out_dir,'aux-{}.npz'.format(scene_id)),**new_aux_data)

    print("sort finished!")

        
        

def gather_sequence_info(detection_file,timestamp_file,bev_size=(200,200)):
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
    if timestamp_file is not None:
        timestamps=np.load(timestamp_file)
    groundtruth = None




    min_frame_idx=int(detections[:,0].min())
    max_frame_idx=int(detections[:,0].max())

    update_ms=100

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    sequence_dir=detection_file.split('.')[0].split('-')[-1]
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "detection_file": detection_file,
        "detections": detections,
        "groundtruth": groundtruth,
        "bev_size": bev_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "timestamp":timestamps
    }
    return seq_info


def create_detections(detection_mat, timestamps,frame_idx,min_height=0):
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
    timestamp_list = timestamps[mask]
    for row in detection_mat[mask]:
        # bbox, confidence, feature = row[2:6], row[6], row[10:]
        det_id,xy, cls,confidence, v_xy,feature = row[1],row[2:4], row[4],row[5],row[6:8], row[8:]

        detection_list.append(DetectionOcc(det_id,xy, cls,confidence,v_xy,feature,timestamp_list[0]))
    return detection_list


def run(detection_file, timestamp_file,aux_file,ori_dir,output_file, min_confidence,max_cosine_distance,nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    timestamp_file : str
        Path to the timestamps file.
    aux_file : str
        Path to the aux file.
    ori_dir : str
        Path to the raw detection result.
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
    seq_info = gather_sequence_info(detection_file,timestamp_file)
    metric_cos = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    metric_geo=1
    tracker = TrackerOcc(metric_cos,metric_geo)
    results = []
    init_flag=False
    cur_timestamp=0
    aux_data=np.load(aux_file)

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)
        print('-'*50)
        nonlocal init_flag
        nonlocal cur_timestamp
        nonlocal aux_data
        nonlocal min_confidence
        file_n=aux_data[str(frame_idx)].item()
        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], seq_info["timestamp"],frame_idx)

        detections = [d for d in detections if d.confidence >= min_confidence]
        
        # Run non-maximum suppression.
        # boxes = np.array([d.tlwh for d in detections])
        centers=np.array([d.xy for d in detections])
        scores = np.array([d.confidence for d in detections])
        # print("min_confidence:{}\tcheck scores:{}".format(min_confidence,scores))
        if not init_flag:
            init_flag=True
            cur_timestamp=detections[0].timestamp
        else:
            raw_det_data=np.load(os.path.join(ori_dir,file_n))
            cur_to_prev_ego=raw_det_data['curr_to_prev_ego_rt'].squeeze()
            prev_to_cur_ego=np.linalg.inv(cur_to_prev_ego)
            tracker.align(prev_to_cur_ego)

        dt=detections[0].timestamp-cur_timestamp
        
        # Update tracker.

        tracker.predict(dt)#extend current track
        match_track_det=tracker.update(detections)#add detection

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
            info = track.to_xycls()
            # results.append([
            #     frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            results.append([
                frame_idx, track.track_id, info[0], info[1], info[2]])#info:cx,cy,cls
        
        frame_match_info=[]
        
        if len(match_track_det)!=0:
            # matches:track,det
            
            for m in match_track_det:
                det_match=m[1]
                track_match=m[0]

                frame_match_info.append([track_match.track_id,track_match.pred[0],track_match.pred[1],track_match.track_cls,det_match.det_id,det_match.xy[0],det_match.xy[1],det_match.cls])
                print("match save:{}\t{}\t cur_det:{}".format(track_match.track_id,det_match.det_id,len(detections)))
            np.save(os.path.join('detections/track_det',file_n.replace('.npz','.npy')),np.asarray(frame_match_info))
            # print("track match:{}".format(info))
            track_info=[]
            for track in tracker.tracks:
                # track_info.append([track.track_id,track.pred[0],track.pred[1],track.track_cls])
                # track_info.append([track.track_id,track.before_align[0],track.before_align[1],track.track_cls])
                track_info.append([track.track_id,track.mean[0],track.mean[1],track.track_cls])
            det_info=[]
            for det in detections:
                det_info.append([det.det_id,det.xy[0],det.xy[1],det.cls])
            np.save(os.path.join('detections/track_mid',file_n.replace('.npz','track.npy')),np.asarray(track_info))
            np.save(os.path.join('detections/track_mid',file_n.replace('.npz','det.npy')),np.asarray(det_info))


    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%d,1,-1,-1,-1' % (
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
    # parser.add_argument(
    #     "--sequence_dir", help="Path to MOTChallenge sequence directory",
    #     default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--timestamp_file", help="Path to timestamps.", default=None,
        required=True)
    parser.add_argument(
        "--aux_file", help="Path to timestamps.", default=None,
        required=True)
    parser.add_argument(
        "--ori_dir", help="Path to raw detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="./results_occ.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.6, type=float)
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
    sort_files('./detections/track_data')
    args = parse_args()
    run(
        args.detection_file, args.timestamp_file,args.aux_file,args.ori_dir,args.output_file,
        args.min_confidence,args.max_cosine_distance, args.nn_budget, args.display)

    