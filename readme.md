## backbone
    yolo是darknet53，use residual block
    poly-yolo是darknet53，use residual block with SE-block，通道数压缩成75%
    本工程提供darknet53, darknet53_se和efficientNet，输出x4-x32的feature

    | params: |
    | ---- | ---- |
    | darknet53  |  42,713,444 |
    | darknet53_se  |  26,704,024 | 
    | efficientNet-b0  |  5,081,328 |


## y_true for current task
    current task: 多类别关键点检测
    [h//4, w//4, 2+1+cls], 2+1+cls for (x,y,conf,cls)
    x,y for normed x,y according to input_shape


## offset tx,ty --- grid coord gx,gy --- normed x,y
    gx = sigmoid(tx) + cx
    gy = sigmoid(ty) + cy
    x = gx / grid_shape_w
    y = gy / grid_shape_h

