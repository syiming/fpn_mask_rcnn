# GSoC 2020 - TenserFlow: Learning Notes

- Learn Google Cloud Platform
    - The restart command is different create command
    - Use version: tf nightly 
    - Don't delete VM
    - Remember to check TPU usage every night and remember to shut it down
        - Use command line to shutdown
        - Use automatica shutdown script
    - Test code using colab resources first, before using TPU resources on GCP
    - Set an alert for account!!!
- Use ReadtheDoc
- **GitHub / Stack Overflow support for Object Detection API**
- Documentation and Colab examples for computer vision (Details to be provided later)
- Blog post





## June

### Task

- Object Detection API
    - Add FPN for Faster RCNN
    - Move all our existing feature extractors to Keras
    - Add Precision/Recall as an eval metric (https://github.com/tensorflow/models/issues/8412)





### FPN: Feature Pyramid Networks for Object Detection

##### Why use FPN?

- The Faster RCNN I learned before only uses the feature comming out from the backbone network. This featurn map contains high semantic information but not much position information. This makes the network having difficulties detecting small objects.

    

    <img src="figure/FasterRCNN.jpg" alt="FasterRCNN" style="zoom:25%;" />



##### FPN idea

<img src="/Users/syiming/Desktop/Memo/figure/FeaturePyramid.png" alt="FeaturePyramid" style="zoom:30%;" />

*Figure 1. (a) Using an image pyramid to build a feature pyramid. Features are computed on each of the image scales independently, which is slow. (b) Recent detection systems have opted to use only single scale features for faster detection. (c) An alternative is to reuse the pyramidal feature hierarchy computed by a ConvNet as if it were a featurized image pyramid. (d) Our proposed Feature Pyramid Network (FPN) is fast like (b) and (c), but more accurate. In this figure, feature maps are indicate by blue outlines and thicker outlines denote semantically stronger features.*

- My understanding: It's a combination of layer with high semantics and layer with accurate position. Thus, it achieves high accuracy while having a resonable running time.



##### FPN Structure

<img src="figure/FPNStructure.png" alt="FPNStructure" style="zoom:50%;" />



### Object Detection API

