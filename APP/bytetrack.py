import numpy as np

class Track:
    def __init__(self, tid, bbox):
        self.track_id = tid
        self.bbox = bbox
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1

    def to_tlwh(self):
        return self.bbox

    def is_confirmed(self):
        return True


class ByteTracker:
    def __init__(self, max_age=15, iou_thresh=0.3):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_thresh = iou_thresh

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1+w1, x2+w2)
        yb = min(y1+h1, y2+h2)

        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0

    def update_tracks(self, detections):
        for t in self.tracks:
            t.predict()

        unmatched = []
        for det in detections:
            x, y, w, h, conf = det
            matched = False

            for t in self.tracks:
                if self.iou((x,y,w,h), t.bbox) > self.iou_thresh:
                    t.update((x,y,w,h))
                    matched = True
                    break

            if not matched:
                unmatched.append(det)

        for det in unmatched:
            x,y,w,h,conf = det
            self.tracks.append(Track(self.next_id, (x,y,w,h)))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        return self.tracks
