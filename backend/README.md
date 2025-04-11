# 视频流处理系统说明文档

## 1. 系统概述

本系统是一个基于WebSocket的视频流处理系统，支持多种视频源的实时处理和转发。系统可以处理无人机和机器人的可见光和热成像视频流，并提供实时处理和转发功能。

## 2. 视频缓存目录

- `drone_visible`: 无人机可见光视频
- `drone_thermal`: 无人机热成像视频
- `robot_visible`: 机器人可见光视频
- `robot_thermal`: 机器人热成像视频

## 3. WebSocket连接

### 3.1 视频源连接
- 连接URL: `ws://{host}:{port}/ws/source`
- 连接成功后，视频源可以发送视频帧数据
- 支持自动保存视频到对应目录

### 3.2 客户端连接
- 连接URL: `ws://{host}:{port}/ws/client`
- 连接成功后，客户端可以请求特定类型的视频流
- 支持实时处理和转发

## 4. 客户端请求格式

### 4.1 请求参数
```json
{
    "device_type": "drone|robot",  // 设备类型
    "video_type": "regular|infrared",  // 视频类型
    "process_type": "original|smoke_removal",  // 处理类型
    "continuous": true,  // 是否持续发送
    "frame_rate": 30  // 期望帧率
}
```

### 4.2 参数说明
- `device_type`: 设备类型，可选值为 "drone" 或 "robot"
- `video_type`: 视频类型，可选值为 "regular" 或 "infrared"
- `process_type`: 处理类型，可选值为 "original" 或 "smoke_removal"
- `continuous`: 是否持续发送视频帧，默认为 true
- `frame_rate`: 期望的帧率，默认为 30fps

## 5. 服务器响应格式

### 5.1 连接成功响应
```json
{
    "status": "connected",
    "message": "Connected to video streaming server"
}
```

### 5.2 可用视频类型响应
```json
{
    "status": "info",
    "message": "Available video types",
    "video_types": ["drone_visible", "drone_thermal", "robot_visible", "robot_thermal"]
}
```

### 5.3 视频帧数据
```json
{
    "device_type": "robot",
    "video_type": "robot_thermal",
    "frame_size": 25964,
    "format": "jpeg",
    "timestamp": 1234567890.123
}
```

### 5.4 错误响应
```json
{
    "status": "error",
    "message": "Error message"
}
```

### 5.5 警告响应（无视频流）
```json
{
    "status": "warning",
    "message": "No video stream available for {video_type}",
    "video_type": "robot_thermal",
    "timestamp": 1234567890.123
}
```

## 6. 视频源断开通知

当视频源断开连接时，所有正在观看该视频流的客户端会收到通知：
```json
{
    "status": "error",
    "message": "Video source disconnected for {video_type}",
    "video_type": "robot_thermal"
}
```

## 7. 视频保存

系统会自动将接收到的视频保存到对应的目录：
- `videos/drone_visible/`: 无人机可见光视频
- `videos/drone_thermal/`: 无人机热成像视频
- `videos/robot_visible/`: 机器人可见光视频
- `videos/robot_thermal/`: 机器人热成像视频

## 8. 错误处理

### 8.1 连接错误
- 当连接断开时，系统会自动清理相关资源
- 客户端会收到相应的错误通知

### 8.2 视频处理错误
- 当视频处理出错时，系统会记录错误日志
- 客户端会收到相应的错误通知

### 8.3 无视频流处理
- 当没有可用的视频流时，系统会定期发送警告通知
- 警告通知间隔为5秒
- 客户端可以根据警告通知更新UI

## 9. 注意事项

1. 视频源连接时，需要先发送设备类型和视频类型信息
2. 客户端连接时，需要先请求可用的视频类型
3. 视频帧数据使用JPEG格式编码
4. 系统会自动处理视频类型的映射（regular -> visible, infrared -> thermal）
5. 建议客户端实现重连机制，以处理网络中断等情况
