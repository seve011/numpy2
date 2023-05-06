import numpy
import torch
import pth_check as p
from uilts import *
from PIL import ImageDraw, ImageFont
import colorsys
import cv2

def zero_pad(X, pad):
    """
    把数据集X的图像边界用0值填充。填充情况发生在每张图像的宽度和高度上。
    
    参数:
    X -- 图像数据集 (m, n_C, n_H, n_W)，分别表示样本数、图像高度、图像宽度、通道数 
    pad -- 整数，每个图像在垂直和水平方向上的填充量
    
    返回:
    X_pad -- 填充后的图像数据集 (m, n_C, n_H + 2*pad, n_W + 2*pad)
    """
    # X数据集有4个维度，填充发生在第2个维度和第三个维度上；填充方式为0值填充
    X_pad = numpy.pad(X, (
        			(0, 0),# 样本数维度，不填充
        			(0, 0), #n_C维度，不填充
        			(pad, pad), #n_W维度，上下各填充pad个像素
        			(pad, pad)), #n_H维度，上下各填充pad个像素
                   mode='constant', constant_values = (0, 0))
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    使用卷积核与上一层的输出结果的一个分片进行卷积运算, only 1 channel
    
    参数:
    a_slice_prev -- 输入分片， (n_C_prev, f, f)
    W -- 权重参数，包含在一个矩阵中 (n_C_prev, f, f)
    b -- 偏置参数

    
    返回:
    Z -- 一个实数，表示在输入数据X的分片a_slice_prev和滑动窗口（W，b）的卷积计算结果
    """

    # 逐元素相乘，结果维度为（f,f,n_C_prev）
    s = numpy.multiply(a_slice_prev, W)
    # 求和
    Z = numpy.sum(s)
    # 加上偏置参数b
    Z = Z + int(b)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    卷积
    
    参数:
    A_prev --- 上一层网络的输出结果，(m, n_C_prev, n_H_prev, n_W_prev)，
    W -- 权重参数，指这一层的卷积核参数 (n_C, n_C_prev, f, f)，n_C个大小为（n_C_prev, f,f）的卷积核
    b -- 偏置参数 (n_C, 1)
    hparameters -- 超参数，包含 "stride" and "pad"
        
    返回:
    Z -- 卷积计算结果，维度为 (m, n_C, n_H, n_W)
    """
    # 输出参数的维度，包含m个样from W's shape (≈1 line)
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    # 权重参数
    (n_C, n_C_prev, f, f) = W.shape
    # 获取本层的超参数：步长和填充宽度
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # 计算输出结果的维度
    # 使用int函数代替np.floor向下取整
    n_H = int((n_H_prev + 2 * pad - f)/stride) + 1
    n_W = int((n_W_prev + 2 * pad - f)/stride) + 1
    
    # 声明输出结果
    Z = numpy.zeros((m, n_C, n_H, n_W))
    
    # 1. 对输出数据A_prev进行0值边界填充
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):  # 依次遍历每个样本
        a_prev_pad = A_prev_pad[i] # 获取当前样本, a_prev_pad:(channel, H, W)
        for c in range(n_C): # 遍历输出的通道
            for h in range(n_H): # 在输出结果的垂直方向上循环
                for w in range(n_W):#在输出结果的水平方向上循环
                    # 确定分片边界
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                
                    # 在输入数据上获取当前切片，结果是3D
                    a_slice_prev = a_prev_pad[:, vert_start:vert_end,horiz_start:horiz_end]
                    # 获取当前的卷积核参数
                    weights = W[c,:,:,:] # weights:(input_channel, H, W)
                    biases = b[c]
                    # 输出结果当前位置的计算值，使用单步卷积函数
                    Z[i, c, h, w] = conv_single_step(a_slice_prev, weights, biases)
                                        
    assert(Z.shape == (m, n_C, n_H, n_W))
    
    return Z

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    池化层的前向传播
    
    参数:
    A_prev -- 输入数据，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    hparameters -- 超参数字典，包含 "f" and "stride"
    mode -- string；表示池化方式, ("max" or "average")
    
    返回:
    A -- 输出结果，维度为 (m, n_C, n_H, n_W)
    """
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    
    f = hparameters["f"] # max_pool size = 2
    stride = hparameters["stride"]
    
    # 计算输出数据的维度
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # 定义输出结果
    A = numpy.zeros((m, n_C, n_H, n_W))              
    # 逐个计算，对A的元素进行赋值
    for i in range(m): # 遍历样本
        for c in range (n_C):# 遍历通道
            for h in range(n_H):# 遍历n_H维度
                # 确定分片垂直方向上的位置
                vert_start = h * stride
                vert_end =vert_start + f
            
                for w in range(n_W):# 遍历n_W维度
                # 确定分片水平方向上的位置
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                
                    # 确定当前样本上的分片
                    a_prev_slice = A_prev[i, c, vert_start:vert_end, horiz_start:horiz_end]
                    # 根据池化方式，计算当前分片上的池化结果
                    if mode == "max":# 最大池化
                        A[i, c, h, w] = numpy.max(a_prev_slice)
                    elif mode == "average":# 平均池化
                        A[i, c, h, w] = numpy.mean(a_prev_slice)
    

    # 确保输出结果维度正确
    assert(A.shape == (m, n_C, n_H, n_W))
    
    return A

def quan(A_prev, gam, M = 13):
    
    """
    反量化与量化处理
    
    参数:
    A_prev -- 输入数据，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    gam -- 反量化参数，维度为(n_C_prev, 1)
    M -- 超参数 为13
    
    返回:
    A -- 输出结果，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    """

    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    
    A = numpy.zeros((m, n_C_prev, n_H_prev, n_W_prev))

    for i in range(m):
        for c in range(n_C_prev):
            A[i][c] = numpy.clip(numpy.rint((A_prev[i][c] * gam[c]) / pow(2, M)).astype(numpy.int16), 0 , 15)    
    
    # 掉点严重
    
    return A.astype(numpy.int16) 

def quan1(A_prev, gam, M = 13):
    
    """
    反量化与量化处理
    
    参数:
    A_prev -- 输入数据，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    gam -- 反量化参数，维度为(n_C_prev, 1)
    M -- 超参数 为13
    
    返回:
    A -- 输出结果，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    """

    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    
    A = numpy.zeros((m, n_C_prev, n_H_prev, n_W_prev))

    for i in range(m):
        for c in range(n_C_prev):
            A[i][c] = numpy.clip(numpy.rint((A_prev[i][c] * gam) / pow(2, M)).astype(numpy.int16), 0 , 15)    
    
    # 掉点严重
    
    return A.astype(numpy.int16) 


def relu(A_prev):
    """
    relu激活处理
    
    参数:
    A_prev -- 输入数据，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    
    返回:
    A -- 输出结果，维度为 (m, n_C_prev, n_H_prev, n_W_prev)
    """
    return numpy.maximum(A_prev, 0)

def test_layer_0(input, weight, bias, gam):
    # input-> conv_forward -> pool_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    pool = pool_forward(conv, {"f":2, "stride":2})
    relu0 = relu(pool)
    return quan(relu0, gam)

def test_layer_2(input, weight, bias, gam):
    # input-> conv_forward -> pool_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    pool = pool_forward(conv, {"f":2, "stride":2})
    relu0 = relu(pool)
    return quan(relu0, gam)

def test_layer_4(input, weight, bias, gam):
    # input-> conv_forward -> pool_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    pool = pool_forward(conv, {"f":2, "stride":2})
    relu0 = relu(pool)
    return quan(relu0, gam)

def test_layer_6(input, weight, bias, gam):
    # input-> conv_forward -> pool_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    pool = pool_forward(conv, {"f":2, "stride":2})
    relu0 = relu(pool)
    return quan(relu0, gam)

def test_layer_8(input, weight, bias, gam):
    # input-> conv_forward -> pool_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    pool = pool_forward(conv, {"f":2, "stride":2})
    relu0 = relu(pool)
    return quan(relu0, gam)

def test_layer_10(input, weight, bias, gam):
    # input-> conv_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    relu0 = relu(conv)
    return quan(relu0, gam)

def test_layer_13(input, weight, bias, gam):
    # input-> conv_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":0})
    relu0 = relu(conv)
    return quan(relu0, gam)

def test_layer_18(input, weight, bias, gam):
    # input-> conv_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":0})
    relu0 = relu(conv)
    return quan(relu0, gam)

def test_layer_19(layer_18, layer_9): #concat
    return numpy.concatenate([layer_18, layer_9], axis=1)
    #增加通道数
    
def test_layer_20(layer19): #upsample
    return numpy.repeat(numpy.repeat(layer19, 2, axis=2), 2, axis=3)
    
def test_layer_21(input, weight, bias, gam):
    # input-> conv_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":1})
    relu0 = relu(conv)
    return quan(relu0, gam)

def test_layer_22(input, weight, bias, gam): #transform channel
    # input-> conv_forward -> relu -> quan
    conv = conv_forward(input, weight, bias, {"stride":1, "pad":0})
    relu0 = relu(conv)
    # print(relu0.shape)
    return quan1(relu0, gam)

def Tiny_yolo3_module(input, weight, bias, gam, gam_shortcut):
    """
    YOLO tiny3 numpy网络(最后一层除外, 共10个conv)
    
    参数:
    photo -- 输入数据, layout = "NCHW"
    weight -- weight, len = 10
    bias -- bias list, len = 9
    gam -- gam list, len = 9
    
    返回:
    A -- 输出结果
    """

    layer_0 = test_layer_0(input, weight[0], bias[0], gam[0])
    layer_2 = test_layer_2(layer_0, weight[1], bias[1], gam[1])
    layer_4 = test_layer_4(layer_2, weight[2], bias[2], gam[2])
    layer_6 = test_layer_6(layer_4, weight[3], bias[3], gam[3])
    layer_8 = test_layer_8(layer_6, weight[4], bias[4], gam[4])
    layer_10 = test_layer_10(layer_8, weight[5], bias[5], gam[5])
    layer_13 = test_layer_13(layer_10, weight[6], bias[6], gam[6])
    layer_18 = test_layer_18(layer_13, weight[7], bias[7], gam[7])
    layer_9 = test_layer_8(layer_6, weight[4], bias[4], gam_shortcut)
    layer_19 = test_layer_19(layer_18, layer_9)
    layer_20 = test_layer_20(layer_19)
    layer_21 = test_layer_21(layer_20, weight[8], bias[8], gam[8])
    layer_22 = test_layer_22(layer_21, weight[9], bias[9], gam[9])

    return layer_22

def detect_process_with_torch(image, weight, bias, gam): #yolo
    
    num_classes = 12
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    input_data, image_shape = process_input(image)

    outputs = Tiny_yolo3_module(input_data, weight, bias, gam)
    
    outputs = torch.from_numpy(outputs)
    ''''
    for i, input in enumerate(outputs):
        print(type(input))
        print(input.shape)
        print(input)
    '''
    outputs = decode_box(outputs, image_shape = [288, 512])
    #---------------------------------------------------------#
    #   将预测框进行堆叠，然后进行非极大抑制
    #---------------------------------------------------------#
    print(type(outputs))
    print(outputs)
    print(torch.cat(outputs, 1))
    results = non_max_suppression(torch.cat(outputs, 1), num_classes = 12, input_shape = [288, 512], image_shape = image_shape, letterbox_image = False, conf_thres = 0.5, nms_thres = 0.3)
    '''                                             
    if results[0] is None: 
        return image
    '''
    print(type(results))
    top_label = numpy.array(results[0][:, 6], dtype = 'int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]
    #---------------------------------------------------------#
    #   设置字体与边框厚度
    #---------------------------------------------------------#
    font = ImageFont.truetype(font='/media/bojiao/9E4D-65EA/numpy/simhei.ttf', size=numpy.floor(3e-2 * image_shape[1] + 0.5).astype('int32'))
    thickness = int(max((image_shape[0] + image_shape[1]) // numpy.mean([288, 512]), 1))
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top = max(0, numpy.floor(top).astype('int32'))
        left = max(0, numpy.floor(left).astype('int32'))
        bottom = min(image_shape[1], numpy.floor(bottom).astype('int32'))
        right = min(image_shape[0], numpy.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)
            
        if top - label_size[1] >= 0:
            text_origin = numpy.array([left, top - label_size[1]])
        else:
            text_origin = numpy.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image

def detect_process_with_numpy(image_path, weight, bias, gam, gam_shortcut): #yolo
    
    num_classes = 12
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    mask = 0, 1, 2
    anchors = 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319

    mask = [int(x) for x in mask]

    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    input_data, image_shape, image = process_input(image_path)

    outputs = Tiny_yolo3_module(input_data, weight, bias, gam, gam_shortcut)

    prediction = predict_transform(outputs, inp_dim_h = 288, inp_dim_w = 512, anchors = anchors, num_classes = num_classes)

    results = non_max_suppression(prediction)

    top_label = numpy.array(results[0][:, 6], dtype = 'int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]
    #---------------------------------------------------------#
    #   设置字体与边框厚度
    #---------------------------------------------------------#
    font = ImageFont.truetype(font='/media/bojiao/9E4D-65EA/numpy/simhei.ttf', size=numpy.floor(3e-2 * image_shape[1] + 0.5).astype('int32'))
    thickness = int(max((image_shape[0] + image_shape[1]) // numpy.mean([288, 512]), 1))
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top = max(0, numpy.floor(top).astype('int32'))
        left = max(0, numpy.floor(left).astype('int32'))
        bottom = min(image_shape[1], numpy.floor(bottom).astype('int32'))
        right = min(image_shape[0], numpy.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)
            
        if top - label_size[1] >= 0:
            text_origin = numpy.array([left, top - label_size[1]])
        else:
            text_origin = numpy.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image


if __name__ == "__main__":
    
    '''
    numpy.random.seed(1)
    
    input = numpy.random.rand(1, 64, 4, 4)
    
    input = (numpy.rint(input * 15)).astype(numpy.uint8)
    '''

    #test for layer 0 to 21
    pthfile = r'/media/bojiao/9E4D-65EA/numpy/lsq_ckpt_94.pth'

    image = '/media/bojiao/9E4D-65EA/numpy/000002.jpg'

    model = torch.load(pthfile, torch.device('cpu'))    
    
    bias, weight, gam, gam_shortcut = p.get_param(model)
    
    result = detect_process_with_numpy(image, weight, bias, gam, gam_shortcut)
    
    result.show()
    
    cv2.waitKey(0)
    '''
    with open(".\layer_0_to_22.txt","w") as outfile:
        outfile.write(str(result))
    '''
    