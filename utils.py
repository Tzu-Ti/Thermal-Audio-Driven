
def detectFace(detector, img):
    img = img.cpu().detach().numpy().transpose(1, 2, 0)

    boxes, scores, classids, kpts = detector.detect(img)
    x, y, w, h = boxes[0].astype(int)
    bigger = 10
    if w >= h: 
        new_h = w
        y_complement = (new_h - h) // 2 + bigger
        x_complement = 0 + bigger
    elif h > w:
        new_w = h
        x_complement = (new_w - w) // 2 + bigger
        y_complement = 0 + bigger
    y1 = y - y_complement
    y2 = y + h + y_complement
    x1 = x - x_complement
    x2 = x + w + x_complement
    face = img[y1: y2, x1: x2, :]
    return [x1, y1, x2, y2], face