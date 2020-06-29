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


## loss
    xy_offset, 相对于grid coords, sigmoid以后位于[0,1]
    conf, sigmoid以后表示posibility, 位于[0,1]
    cls, sigmoid以后表示posibility, 位于[0,1]
    所以三个loss都是bce，我刚开始conf_loss用了focal loss，网络很快收敛但是效果巨差，
    换成bce以后其他两个loss仍旧很小，conf_loss变得贼大，所以focal loss可能写错了
    focal loss果然写错了。。。FL=-alpha * (1-pt)^gamma * log(pt)


## todolist
    8倍下采样
    check focal loss


