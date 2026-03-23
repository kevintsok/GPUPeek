# CUDA Graph Research

## 概述

CUDA Graph 通过图捕获、实例化和启动来减少内核启动开销。

## 1. 工作流程

```
Capture → Instantiate → Launch
```

## 2. Capture

```cuda
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel<<<...>>>(...);  // 捕获
cudaStreamEndCapture(stream, &graph);
```

## 3. Instantiate

```cuda
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```

## 4. Launch

```cuda
cudaGraphLaunch(graphExec, stream);
```

## 5. 优势

- 减少内核启动开销
- 更稳定的延迟
- 适合批量处理

## 参考文献

- [CUDA Programming Guide - Graphs](../ref/cuda_programming_guide.html)
