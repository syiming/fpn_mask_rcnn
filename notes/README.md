# GSoC 2020 - TenserFlow: Learning Notes

- Learn Google Cloud Platform
    - https://gcp-tutorial.readthedocs.io/en/latest/index.html
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

        - June 9: ~~Download dataset and try to train a dataset~~
            -  <span style="color:red">When using runtime-version 1.12 encounter problem: tf.compat has no module v1</span>: Fixed using runtime version 1.15
            - <span style="color:red">Cannot allocate memory at iteration 3 or 4</span>
        - June 10-13: try to implement FPN 

            - 10: ~~Find the entry point and setup implemetation plan~~
        - 11: ~~Exam on June 11 PST. Possible canceling thursday meeting?~~
            - 12-14: 
        - implementation tips:
                - Follow code structure (below)
                - target assignment need to adjust to multi feature (faster rcnn meta arch)
                    - fork / open a pull request / push small piece
            - test every step
            - 15:

                - setup training loop: overfit on small data
            - test

```python
# possible code structure
# in models/faster_rcnn_fpn_keras_feature_extractor.py

# No need to implement this function
# Fpn_feature_levels (https://github.com/tensorflow/models/blob/master/research/object_detection/utils/ops.py#L1009)
# def assign_feature_levels():
#   # TODO: assignment strategy of region-based detectors
#   # if the RoI’s scale becomes smaller, it should be mapped into a finer-resolution level
#   # $$k=\lfloor k_0 + \log_2(\sqrt{wh}/224)\rfloor$$ 
#   # input box size, k_0
#   # output k
#   pass

class FasterRCNNFPNKerasFeatureExtractor(faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  def __init__(self, ...):
    # TODO: constructor
    pass
  
  def build:
    # TODO: Build the structure, should be very similar as ssd_*_fpn_keras_feature_extractor.py
    # ResNet-101 (object_detection.models.keras_models)
    # object_detection.models.feature_map_generators
    pass
  
  def preprocess(self, ...):
    # TODO: should be the same as others
    pass
  
  def _extract_proposal_features(self, ...):
    # TODO: Extracts first stage RPN features
    # [x] assign_feature_levels() 
    # 	⬆️ Fpn_feature_levels 
    # [x] spatial_transform_ops.multilevel_roi_align
    # 	⬆️ is in faster rcnn meta arch
    pass
  
  def _extract_box_classifier_features(self, ...):
    # TODO: Extracts second stage box classifier features.
    pass
  
  def restore_from_classification_checkpoint_fn(self, ...):
    # follow the none fpn version
    pass
    

# predictor using convolutional keras box predictor
```



- Move all our existing feature extractors to Keras
- Add Precision/Recall as an eval metric (https://github.com/tensorflow/models/issues/8412)



---



### FPN: Feature Pyramid Networks for Object Detection

##### Why use FPN?

- The Faster RCNN I learned before only uses the feature comming out from the backbone network. This featurn map contains high semantic information but not much position information. This makes the network having difficulties detecting small objects.

    

    <img src="figure/FasterRCNN.jpg" alt="FasterRCNN" style="zoom:25%;" />



##### FPN idea

<img src="figure/FeaturePyramid.png" alt="FeaturePyramid" style="zoom:30%;" />

*Figure 1. (a) Using an image pyramid to build a feature pyramid. Features are computed on each of the image scales independently, which is slow. (b) Recent detection systems have opted to use only single scale features for faster detection. (c) An alternative is to reuse the pyramidal feature hierarchy computed by a ConvNet as if it were a featurized image pyramid. (d) Our proposed Feature Pyramid Network (FPN) is fast like (b) and (c), but more accurate. In this figure, feature maps are indicate by blue outlines and thicker outlines denote semantically stronger features.*

- My understanding: It's a combination of layer with high semantics and layer with accurate position. Thus, it achieves high accuracy while having a resonable running time.



##### FPN Structure

<img src="figure/FPNStructure.png" alt="FPNStructure" style="zoom:50%;" />

##### For Faster RCNN: assign RoI of different scales to the pyramid levels

Assign an RoI of width $w$ and height $h$ (on the input image to the network) to the level $P_k$ of feature pyramid by:

$$k=\lfloor k_0 + \log_2(\sqrt{wh}/224)\rfloor$$

Here 224 is the canonical ImageNet pre-training size, and $k_0$ is the target level on which an RoI with $w×h = 224^2$ should be mapped into.



---



### Object Detection API

#### Meta Architectures

- Detection Model Interface: core/model.py:
    - tensor-in / tensor-out design
    - computation graphs
        - Training time:
            inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)
        - Evaluation time:
            inputs (images tensor) -> preprocess -> predict -> postprocess
             -> outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)
    - Notes:
        - standard_fields: naming convention
        - DetectionModel: a class containing many abstact methods.
            - get_side_inputs (static): 
            - preprocess: 
                - scaling, shifting, resizing, padding
                - no trainable parameters
                - batchsize != batchsize in predict (assuming 1)
                - return processed image and original shape
                -  <font color='red'>There is also no explicit assumption that the output resolutions must be fixed across inputs --- this is to support *"fully convolutional" settings* in which input images can have different shapes/resolutions. ??? fully convolutional neural network</font>
            - predict:
                - input processed image and original shape
                - output prediction
            - postprocess:
                - get final detection
                - NMS / convert score / remove backgroud class
                - Output
                    - Class: [0, num_classes)
                    - Boxes: [y_min, x_min, y_max, x_max]
                    - detections can pad into fixed number of boxes
                    - <font color='red'>What is keypoints?</font>
            - loss:
                - need group truth
            - provide_groundtruth:
                - add groudtruth
                    - boxes and classes ground truth
            - regularization_losses:
            - restore_map:
                - restore from check point
            - updates:
                - <font color='red'>Which operators...?</font>
                    - meta 里面有没有update / not used in TF2
            - call:
                - connect all steps



#### Code Pointers

##### Data pipeline

- inputs.py
    - train_input:
        - output features and labels
        - data_augmentation
        - image_resizer
        - transform_input_data: use data_augmentation & image_resizer
        - pad_input_data_to_static_shapes: use transform_input_data
        - INPUT_BUILDER_UTIL_MAP['dataset_build']: use pad_input_data_to_static_shapes

#####  Reader

- core/data_decoder.py:
    - DataDecoder (abstract):
        - return dictionary associate image and labels.

##### Data Augmenter

- core/preprocessor.py:

    - preprocess

        - input tensor, list of tuple (function, dict of args), map function with args, cache (deterministic using same cache).

        

#### Target Assignment

- target_assigner.py:
    - compute similarity
    - compute matching
    - assign regression targets: use matching, BoxCoder
    - assign classification targets: use matching
    - one image a time
- core/matcher.py
- core/region_similarity_calculator.py
- core/boxCoder



#### Prediction

##### Feature Extractor

- meta_architectures/faster_rcnn_meta_arch.py:
    - three modes: number of stages different {1, 2, 3}
        - 1 stage: only RPN
        - 2 stages: proposals -> box classifier
        - 3 stages: eval: proposals -> box classifier -> NMS -> box classifier
    - functions overide:
        - preprocess
        - _extract_proposal_features
        - _extract_box_classifier_features

##### Head

- predictors/heads/head.py:
    - head class
    - overide predict
    - add to meta structure
    - compute loss
    - (optional) visualization

##### Box Predictor

- core/box_predictor.py：
    - passed directly to loss functions
    - predict location, predict class

##### Anchor Generator

- core/anchor_generator.py
    - create (or load) a collection of bounding boxes



















