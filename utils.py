import torch
import torchvision.transforms.functional as TF

def detectFace(detector, imgs, bigger=10, size=96):
    # imgs: tensor of shape (B, 3, H, W)
    batch_boxes = detector.face_detector.detect_from_batch(imgs*255.0)
    coords = []
    faces = []
    for boxes, img in zip(batch_boxes, imgs):
        x1, y1, x2, y2, score = boxes[0].astype(int)
        w = x2 - x1
        h = y2 - y1
        # let bounding box to a square
        if w > h:
            y1 -= (w - h) // 2
            y2 += (w - h) // 2
        else:
            x1 -= (h - w) // 2
            x2 += (h - w) // 2
        x1 -= bigger
        y1 -= bigger
        x2 += bigger
        y2 += bigger
        coords.append([x1, y1, x2, y2])
        face = img[:, y1: y2, x1: x2]
        face = TF.resize(face, [size, size]).unsqueeze(0)
        faces.append(face)
    faces = torch.cat(faces, dim=0) # B, C, H, W
        
    return coords, faces