# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    

class DetectionOcc(object):
    """
    This class represents a heatmap detection in single bev.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, xy, cls,confidence,v_xy,feature,timestamp):
        self.xy = np.asarray(xy, dtype=np.float32)
        self.cls=cls
        self.confidence = float(confidence)
        self.v_xy=np.asarray(v_xy,dtype=np.float32)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.timestamp=timestamp

    # def to_tlbr(self):
    #     """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    #     `(top left, bottom right)`.
    #     """
    #     ret = self.tlwh.copy()
    #     ret[2:] += ret[:2]
    #     return ret

    # def to_xyah(self):
    #     """Convert bounding box to format `(center x, center y, aspect ratio,
    #     height)`, where the aspect ratio is `width / height`.
    #     """
    #     ret = self.tlwh.copy()
    #     ret[:2] += ret[2:] / 2
    #     ret[2] /= ret[3]
    #     return ret

    def to_xywithvel(self):
        ret =np.concatenate([self.xy.copy(),self.v_xy.copy()],axis=-1)
        return ret


