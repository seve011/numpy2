import numpy as np
from PIL import Image
import torch
from torch import Tensor

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

def process_input(image_path):
    image = Image.open(image_path)
    image_shape = np.array(np.shape(image)[0:2])
    input_shape = [288, 512]
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
    image = cvtColor(image)
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    image_data  = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image = False)
    #---------------------------------------------------------#
    #   添加上batch_size维度
    #---------------------------------------------------------#
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    return image_data, image_shape, image

def decode_box(data, input_shape):
    outputs = []
    num_classes = 12
    bbox_attrs = 5 + num_classes

    mask = 0, 1, 2
    anchors = 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319
    mask = [int(x) for x in mask]

    anchors = [int(a) for a in anchors]
    anchors = [(anchors[m], anchors[m + 1]) for m in range(0, len(anchors), 2)]
    anchors = [anchors[n] for n in mask]

    input = data
    #-----------------------------------------------#
    #   输入的input一共有1个，shape是
    #   batch_size, 3*( num_classes + 5), 26, 26, 通常batch_size为1
    #-----------------------------------------------#
    batch_size      = input.size(0)
    input_height    = input.size(2)
    input_width     = input.size(3)

    #-----------------------------------------------#
    #   输入为416x416时
    #   stride_h = stride_w = 16
    #-----------------------------------------------#
    stride_h = input_shape[0] / input_height
    stride_w = input_shape[1] / input_width
    #-------------------------------------------------#
    #   此时获得的scaled_anchors大小是相对于特征层的
    #-------------------------------------------------#

    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in (anchors[anchor_mask] for anchor_mask in mask)]
        
    prediction = input.view(batch_size, len(mask), bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

    #-----------------------------------------------#
    #   先验框的中心位置的调整参数
    #-----------------------------------------------#
    x = torch.sigmoid(prediction[..., 0])  
    y = torch.sigmoid(prediction[..., 1])
    #-----------------------------------------------#
    #   先验框的宽高调整参数
    #-----------------------------------------------#
    w = prediction[..., 2]
    h = prediction[..., 3]
    #-----------------------------------------------#
    #   获得置信度，是否有物体
    #-----------------------------------------------#
    conf = torch.sigmoid(prediction[..., 4])
    #-----------------------------------------------#
    #   种类置信度
    #-----------------------------------------------#
    pred_cls = torch.sigmoid(prediction[..., 5:])

    FloatTensor =  torch.FloatTensor
    LongTensor  =  torch.LongTensor

    #----------------------------------------------------------#
    #   生成网格，先验框中心，网格左上角 
    #   batch_size,3,26,26
    #----------------------------------------------------------#
    grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        batch_size * len(mask), 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
        batch_size * len(mask), 1, 1).view(y.shape).type(FloatTensor)

    #----------------------------------------------------------#
    #   按照网格格式生成先验框的宽高
    #   batch_size,3,26,26
    #----------------------------------------------------------#
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

    #----------------------------------------------------------#
    #   利用预测结果对先验框进行调整
    #   首先调整先验框的中心，从先验框中心向右下角偏移
    #   再调整先验框的宽高。
    #----------------------------------------------------------#
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    #----------------------------------------------------------#
    #   将输出结果归一化成小数的形式
    #----------------------------------------------------------#
    _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
    output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), -1)
    outputs.append(output.data)

    return outputs

def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset = (input_shape - new_shape)/2./input_shape
        scale = input_shape/new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def non_max_suppression_with_torch(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 17]
    #----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
                
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
                
            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
        if output[i] is not None:
            output[i] = output[i].cpu().np()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
    
def predict_transform(prediction, inp_dim_h, inp_dim_w, anchors, num_classes):
    [batch_size, _, grid_size_h, grid_size_w] = np.shape(prediction)
    stride = int(inp_dim_h / grid_size_h)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    grid_x = np.repeat(np.arange(grid_size_w).reshape(1, grid_size_w), grid_size_h, axis=0)
    grid_y = np.repeat(np.arange(grid_size_h).reshape(grid_size_h, 1), grid_size_w, axis=1)
    scaled_anchors = np.array([[a_w / stride, a_h / stride] for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].reshape(3)
    anchor_h = scaled_anchors[:, 1:2].reshape(3)

    prediction = prediction.reshape(batch_size, num_anchors, bbox_attrs, grid_size_h, grid_size_w).transpose(0, 1, 3, 4,
                                                                                                             2)
    pre_t = np.zeros([batch_size, 5, bbox_attrs])
    for i_img in range(batch_size):
        i_pre = prediction[i_img]
        indexs = np.argwhere(i_pre[..., 4] >= 0)
        n_pre = np.shape(indexs)[0]
        n = np.min(np.array([n_pre, 5]))
        for i in range(n):
            index0 = indexs[i][0]
            index1 = indexs[i][1]
            index2 = indexs[i][2]
            pre_t[i_img, i, 0] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 0])) + grid_x[index1, index2]
            pre_t[i_img, i, 1] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 1])) + grid_y[index1, index2]
            pre_t[i_img, i, 2] = np.exp(i_pre[index0, index1, index2, 2]) * anchor_w[index0]
            pre_t[i_img, i, 3] = np.exp(i_pre[index0, index1, index2, 3]) * anchor_h[index0]
            pre_t[i_img, i, 4:] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 4:]))

    pre_t[:, :, :4] *= stride
    return pre_t


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x-
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y-
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x+
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y+
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # get the corrdinates of the intersection rectangle
    x1 = np.concatenate((np.repeat(b1_x1, np.size(b2_x1), axis=0).reshape(1, -1), b2_x1.reshape(1, -1)), axis=0)
    inter_rect_x1 = np.max(x1, 0)
    y1 = np.concatenate((np.repeat(b1_y1, np.size(b2_y1), axis=0).reshape(1, -1), b2_y1.reshape(1, -1)), axis=0)
    inter_rect_y1 = np.max(y1, 0)
    x2 = np.concatenate((np.repeat(b1_x2, np.size(b2_x2), axis=0).reshape(1, -1), b2_x2.reshape(1, -1)), axis=0)
    inter_rect_x2 = np.min(x2, 0)
    y2 = np.concatenate((np.repeat(b1_y2, np.size(b2_y2), axis=0).reshape(1, -1), b2_y2.reshape(1, -1)), axis=0)
    inter_rect_y2 = np.min(y2, 0)
    # Intersection area
    inter_w = np.zeros([2, np.size(inter_rect_x1)])
    inter_w[0] = inter_rect_x2 - inter_rect_x1 + 1
    inter_h = np.zeros([2, np.size(inter_rect_y1)])
    inter_h[0] = inter_rect_y2 - inter_rect_y1 + 1
    inter_area = np.max(inter_w, 0) * np.max(inter_h, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]  # Object confidence filtering

        # If none are remaining => process next image
        if not np.size(image_pred):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        n = np.shape(image_pred)[0]
        class_confs = np.max(image_pred[:, 5:], 1).reshape(n, 1)
        class_preds = np.zeros([n, 1])
        for i in range(n):
            class_preds[i] = np.where(image_pred[i, 5:] == np.max(image_pred[i, 5:]))[0][0]

        detections = np.concatenate((image_pred[:, :5], class_confs, class_preds), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while np.shape(detections)[0]:
            large_overlap = bbox_iou(detections[0:1, :4], detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = np.sum(weights * detections[invalid, :4], 0) / np.sum(weights)
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            n_box = np.shape(keep_boxes)[0]
            output[image_i] = np.zeros([n_box, 7])
            for i in range(n_box):
                output[image_i][i] = keep_boxes[i]

    return output

def class_num2name(num):
    return str(int(num))