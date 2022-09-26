[R-CNN](https://arxiv.org/pdf/1311.2524.pdf)

Image Input -> Extract Region Proposal(~2k) -> Use of warped region(fixed image size) -> Compute CNN features  -> Classify regions

1. Use of selective search(heuristic) to create region proposal around 2k proposals from an image.
2. Use Alex Net as feature extractor (4096-dimension)on warped region coming out of region proposal.
3. Classify region using Class specific Linear SVM with 21-way classification layer.



Notes:

1. At inference time, use of greedy non maximum suppression 
   1. For each class independently reject a region if it has an IOU overlap with a higher scoring(svm score) selected region larger than a learned threshold.
2. Training Objective

   1. Finetune Feature Extractor on domain specific task.
      1. Add a 21 class classification layer(Pascal VOC) in the AlexNet.
      2. Treat all regions with >=0.5 IOU overlap with the ground truth box as positives for that box's class and rest as negatives.

   1. Train SVM classifier on domain specific task.

      1. Training Sample selection
         1. Any proposal with IOU<0.3 with GT is a negative sample.
         2. Proposal with extact GT overlap is a positive example.
         3. Anything between them is discarded.


[Spatial Pyramid Pooling](https://www.youtube.com/watch?v=v3jryjHk820&list=PLoEMreTa9CNm18TPHIYm3t2CLIqxLxzYD&index=5)

1. Instead of subsampling the image first and then passing the region proposals through CNN, we pass the input image through the CNN and then sub sample the feature map using **different** window size and then concat them together to form a fixed size representation to pass it to the fully connected layer.



[Fast RCNN](https://www.youtube.com/watch?v=GRWmdfX9JqM&list=PLoEMreTa9CNm18TPHIYm3t2CLIqxLxzYD&index=7)

1. Concept of ROI Pooling Layer
   1. Bbox of original image is projected on the feature map giving region of interest and it is sub sampled only **once** using a window size and then max pooling each bin to get a fixed size representation feature map.
   2. Use of Classification Head and Bbox regressor as the output of the network instead of svm classifier.
   3. Similar to Spatial Pyramid Pooling except use of single window size to create multiple bins in the feature map.



Faster RCNN

1. Use of Region Proposal Network(RPN) a fully convolution network that simultaneously predicts object bounds and objectness scores at each position.

2. Use of 9 anchors of sizes 128x128,256x256,512x512 with 3 different aspect ratios(1:1,1:2,2:1) that created proposal at each pixel in the feature map.

   Feature map with WxH will generate WxHx9 proposals.

3. Training Process

   1. To be added

4. GT creation

   1. To be added



R-FCN (Region Based Fully Convolutional Network)

1. Maintained translation variance in object detection

Feature Pyramid Network