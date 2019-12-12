import cv2 
import numpy as np

def iou_calculate(box1, box2):
    '''
    calculate iou of 'box1' and 'box2'
    param box1, box2: cx, cy, w, h
    return iou
    '''
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    x1_min, x1_max = box1[0] - int(box1[2]/2), box1[0] + int(box1[2]/2)
    x2_min, x2_max = box2[0] - int(box2[2]/2), box2[0] + int(box2[2]/2)
    y1_min, y1_max = box1[1] - int(box1[3]/2), box1[1] + int(box1[3]/2)
    y2_min, y2_max = box2[1] - int(box2[3]/2), box2[1] + int(box2[3]/2)
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    if x_min < x_max and y_min < y_max:
        area_overlap = (x_max - x_min) * (y_max - y_min)
    else:
        area_overlap = 0 

    iou = area_overlap * 1.0 / (area1 + area2 - area_overlap)
    return iou

def iou_rotate_calculate(boxes1, boxes2):
    '''
    calculate iou of 'rotate boxes1' and 'rotate boxes2'
    param boxes1, boxes2: cx, cy, w, h, theta
    return iou 
    '''
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    # print(int_pts)
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        # print(order_pts)
        int_area = cv2.contourArea(order_pts)
        # print(int_area)
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious=0
    return ious

def plane_detection_validation(list_anno, list_pred, threshold=0.5):
    '''
    valid detection.
    param list_anno: boxes like [class, xmin, ymin, xmax, ymax]
    param list_pred: boxes like [class, confidence, xmin, ymin, xmax, ymax]
    param threshold: iou threshold
    return plane_num(int)
    return right_detect_boxes: boxes like [cx, cy, w, h]
    return wrong_detect_boxes: boxes like [cx, cy, w, h]
    return miss_detect_boxes: boxes like [cx, cy, w, h]
    '''
    plane_num = 0
    right_detect_boxes = []
    wrong_detect_boxes = []
    miss_detect_boxes = [] 
    pred_used = [False for _ in range(len(list_pred))]
    for _, xmin, ymin, xmax, ymax in list_anno:
        w, h = (int(xmax) - int(xmin)), (int(ymax) - int(ymin))
        cx, cy = int(xmin) + int(w/2), int(ymin) + int(h/2)
        box1 = [cx, cy, w, h]
        max_iou = -1
        max_iou_ind = -1
        for i in range(len(pred_used)):
            if pred_used[i]:
                continue
            _, _, xmin, ymin, xmax, ymax = list_pred[i]
            w, h = (int(xmax) - int(xmin)), (int(ymax) - int(ymin))
            cx, cy = int(xmin) + int(w/2), int(ymin) + int(h/2)
            box2 = [cx, cy, w, h]
            iou = iou_calculate(box1, box2)
            if iou > threshold and iou > max_iou:
                max_iou_ind = i 
                max_iou = iou 
        # miss
        if max_iou_ind == -1:
            miss_detect_boxes.append(box1)
        # right
        else:
            pred_used[max_iou_ind] = True 
            _, _, xmin, ymin, xmax, ymax = list_pred[max_iou_ind]
            w, h = (int(xmax) - int(xmin)), (int(ymax) - int(ymin))
            cx, cy = int(xmin) + int(w/2), int(ymin) + int(h/2)
            box2 = [cx, cy, w, h]
            right_detect_boxes.append(box2)
        plane_num += 1 
    # wrong
    for i in range(len(pred_used)):
        if pred_used[i] == False:
            _, _, xmin, ymin, xmax, ymax = list_pred[i]
            w, h = (int(xmax) - int(xmin)), (int(ymax) - int(ymin))
            cx, cy = int(xmin) + int(w/2), int(ymin) + int(h/2)
            box2 = [cx, cy, w, h]
            wrong_detect_boxes.append(box2)
    return plane_num, right_detect_boxes, wrong_detect_boxes, miss_detect_boxes

def get_plane_info(anno_npy_path, pred_npy_path):
    anno = np.load(anno_npy_path) 
    pred = np.load(pred_npy_path)
    plane_num, right, wrong, miss = plane_detection_validation(anno, pred, 0.3)
    print('Total plane number:', plane_num)
    print('Right:', len(right))
    print('Wrong:', len(wrong))
    print('Miss:', len(miss))

def ship_detection_validation(image_name, list_anno, list_pred, threshold=0.5):
    '''
    valid detection.
    param image_name: image path
                      h, w, _ = image.shape
                      if max(h, w) < 2000 
                        -> 
                          [x1, y1, x2, y2, x3, y3, x4, y4] 
                        = [x1, y1, x2, y2, x3, y3, x4, y4] * (min(h, w) / 832)
    param list_anno: boxes like [class, x1, y1, x2, y2, x3, y3, x4, y4, 
                                 cx, cy, w, h, delta]
    param list_pred: boxes like [class, confidence, x1, y1, x2, y2, x3, y3, x4, y4, 
                                 cx, cy, w, h, delta]
    param threshold: iou threshold
    return ship_num(int)
    return right_detect_boxes: boxes like [x1, y1, x2, y2, x3, y3, x4, y4]
    return wrong_detect_boxes: boxes like [x1, y1, x2, y2, x3, y3, x4, y4]
    return miss_detect_boxes: boxes like [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    image = cv2.imread(image_name)
    h, w, _ = image.shape 
    if max(h, w) < 2000:
        num = min(h, w) / 832
    else:
        num = 1
    ship_num = 0
    right_detect_boxes = []
    wrong_detect_boxes = []
    miss_detect_boxes = [] 
    pred_used = [False for _ in range(len(list_pred))]
    for _, x1, y1, x2, y2, x3, y3, x4, y4, cx, cy, w, h, delta in list_anno:
        four_points = [x1, y1, x2, y2, x3, y3, x4, y4]
        cx, cy, w, h, delta = int(cx), int(cy), int(w), int(h), int(delta)
        box1 = [cx, cy, w, h, delta]
        max_iou = -1
        max_iou_ind = -1
        for i in range(len(pred_used)):
            if pred_used[i]:
                continue
            _, _, _, _, _, _, _, _, _, _, cx, cy, w, h, delta = list_pred[i]
            cx, cy, w, h, delta = int(cx) * num, int(cy) * num, int(w) * num, int(h) * num, int(delta) * num
            box2 = [cx, cy, w, h, delta]
            iou = iou_rotate_calculate(box1, box2)
            if iou > threshold and iou > max_iou:
                max_iou_ind = i 
                max_iou = iou 
        # miss
        if max_iou_ind == -1:
            miss_detect_boxes.append(four_points)
        # right
        else:
            pred_used[max_iou_ind] = True 
            _, _, x1, y1, x2, y2, x3, y3, x4, y4, _, _, _, _, _ = list_pred[max_iou_ind]
            box2 = [x1, y1, x2, y2, x3, y3, x4, y4]
            right_detect_boxes.append(box2)
        ship_num += 1 
    # wrong
    for i in range(len(pred_used)):
        if pred_used[i] == False:
            _, _, x1, y1, x2, y2, x3, y3, x4, y4, _, _, _, _, _ = list_pred[i]
            box2 = [x1, y1, x2, y2, x3, y3, x4, y4]
            wrong_detect_boxes.append(box2)
    return ship_num, right_detect_boxes, wrong_detect_boxes, miss_detect_boxes

def get_ship_info(image_name, anno_npy_path, pred_npy_path):
    anno = np.load(anno_npy_path) 
    pred = np.load(pred_npy_path)
    plane_num, right, wrong, miss = ship_detection_validation(image_name, anno, pred, 0.5)
    print('Total plane number:', plane_num)
    print('Right:', len(right))
    print('Wrong:', len(wrong))
    print('Miss:', len(miss))


if __name__ == '__main__':
    # boxes1 = np.array([50, 50, 100, 100, 0], np.float32)
    # boxes2 = np.array([50, 50, 100, 100, -30.], np.float32)
    # iou = iou_rotate_calculate(boxes1, boxes2)
    # print(iou)

    # box1 = np.array([50, 50, 100, 200], np.float32)
    # box2 = np.array([50, 50, 100, 100], np.float32)
    # iou = iou_calculate(box1, box2)
    # print(iou)
    get_plane_info('./plane_anno.npy', './plane_pred.npy')
    get_ship_info('s01.png', './ship_anno.npy', './ship_pred.npy')
