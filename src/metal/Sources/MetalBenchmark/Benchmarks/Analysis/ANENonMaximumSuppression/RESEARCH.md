# ANE Non-Maximum Suppression Research

## Overview

Non-Maximum Suppression (NMS) is a post-processing algorithm used in
object detection to eliminate overlapping bounding boxes. Given a set of
detections, NMS keeps the box with highest confidence and removes all
boxes that have high overlap (IoU > threshold) with it.

## Algorithm

```
1. Sort boxes by confidence score (descending)
2. While boxes remain:
   a. Take highest scoring box
   b. Remove all boxes with IoU > threshold
   c. Add selected box to output
3. Return kept boxes
```

## Complexity

- Time: O(n^2) where n = number of boxes
- Space: O(n) for storing indices
- Sequential by nature - hard to parallelize

## Applications

1. Object Detection (YOLO, SSD, Faster R-CNN)
2. Face Detection
3. Instance Segmentation
4. Video Object Tracking
5. Pedestrian Detection

## Benchmark Results

### Box Count Scaling (IoU = 0.5)
| Box Count | CPU Time (ms) | GPU Time (ms) | Keep Rate |
|-----------|---------------|---------------|-----------|
| 100 | 39.47 | 9.03 | 85.0% |
| 500 | 372.09 | 19.76 | 62.0% |
| 1000 | 1044.71 | 32.26 | 50.7% |
| 2000 | 2456.93 | 36.58 | 37.4% |
| 5000 | 8469.31 | 83.94 | 25.4% |


### IoU Threshold Impact (1000 boxes)
| Threshold | CPU Time (ms) | GPU Time (ms) | Keep Rate |
|------------|---------------|---------------|-----------|
| 0.3 | 336.93 | 16.14 | 20.3% |
| 0.4 | 552.45 | 15.53 | 31.7% |
| 0.5 | 953.60 | 15.67 | 46.9% |
| 0.6 | 1514.50 | 18.27 | 69.0% |
| 0.7 | 2040.28 | 22.86 | 85.3% |
| 0.9 | 2529.77 | 35.78 | 100.0% |


### Real-Time Feasibility
| Objects | Image Size | CPU (ms) | GPU (ms) | FPS |
|---------|------------|----------|----------|-----|
| 10 | 416x416 | 8.97 | 0.93 | 111 |
| 50 | 416x416 | 12.30 | 0.83 | 81 |
| 100 | 640x640 | 12.28 | 0.65 | 81 |
| 100 | 1080x1080 | 10.48 | 0.71 | 95 |
| 200 | 640x640 | 12.70 | 0.56 | 79 |
| 300 | 1920x1920 | 33.37 | 0.52 | 30 |


## Key Insights

1. **NMS is the bottleneck**: With dense object detection, NMS can take
   more time than the detection itself
2. **Parallelization is hard**: The sequential nature of suppression
   makes GPU acceleration limited
3. **Confidence filtering helps**: Pre-filtering low-confidence boxes
   before NMS significantly improves speed
4. **IoU threshold trade-off**: Lower threshold = more suppression but
   slower; higher threshold = faster but more duplicates

## Optimization Strategies

1. **Pre-filtering**: Remove boxes below confidence threshold before NMS
2. **Soft-NMS**: Instead of removing, reduce confidence of overlapping boxes
3. **Multi-scale NMS**: Apply NMS at each feature pyramid level separately
4. **Batch NMS**: Process multiple images in parallel when available

## ANE Suitability

NMS is NOT well-suited for ANE because:
- ANE is optimized for parallel neural network inference
- NMS has sequential dependencies (can't process boxes independently)
- GPU with warp-level parallelism is better suited

However, ANE can accelerate:
- Object detection backbone (ResNet, MobileNet)
- Feature extraction for box generation
- Confidence scoring networks

## Future Work

- Implement Soft-NMS variants
- Compare with learned NMS approaches
- Study the impact of box aspect ratios
- Analyze NMS for rotated bounding boxes