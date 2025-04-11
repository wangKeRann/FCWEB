# WebSocket 视频流请求协议

## 连接信息

- WebSocket 服务器地址: `ws://localhost:8000/ws/client`
- 连接建立后自动发送视频流请求
- 支持自动重连机制（断开后5秒自动重连）

## 请求格式

前端在建立连接后，会发送如下格式的 JSON 请求：

```json
{
    "device_type": "drone" | "robot",  // 设备类型：无人机或机器人
    "video_type": "regular" | "infrared",  // 视频类型：可见光或热成像
    "process_type": "original" | "smoke_removal"  // 处理类型：原始视频或去雾处理
}
```

### 参数说明

1. `device_type`（设备类型）：
   - `"drone"` - 无人机视频
   - `"robot"` - 机器人视频

2. `video_type`（视频类型）：
   - `"regular"` - 可见光视频
   - `"infrared"` - 热成像视频

3. `process_type`（处理类型）：
   - `"original"` - 原始视频（默认）
   - `"smoke_removal"` - 去雾处理后的视频

## 响应格式

### 1. 状态消息（JSON格式）

```json
{
    "status": "warning" | "error",
    "message": "状态描述信息"
}
```

状态类型：
- `"warning"` - 警告信息（如无视频流）
- `"error"` - 错误信息

### 2. 视频数据（二进制格式）

- 视频数据以二进制形式发送
- 视频编码格式：MP4 (codec: avc1.42E01E, mp4a.40.2)
- 视频数据会实时追加到 MediaSource 中播放

## 请求示例

1. 请求无人机可见光视频：
```json
{
    "device_type": "drone",
    "video_type": "regular",
    "process_type": "original"
}
```

2. 请求机器人热成像视频（带去雾处理）：
```json
{
    "device_type": "robot",
    "video_type": "infrared",
    "process_type": "smoke_removal"
}
```

## 状态处理

1. 连接状态：
   - 连接成功：发送视频流请求
   - 连接断开：5秒后自动重连
   - 连接错误：显示错误状态

2. 视频流状态：
   - 正常：持续接收并播放视频数据
   - 无视频流：显示"无视频流"提示
   - 错误：显示错误信息

## 注意事项

1. 所有请求参数都是必需的
2. 参数值必须完全匹配上述选项
3. 服务器需要正确处理以下情况：
   - 连接建立时的初始请求
   - 设备切换时的请求更新
   - 模式切换时的请求更新
   - 无视频流时的状态通知
   - 错误情况的状态通知

4. 视频流处理：
   - 视频数据需要实时发送
   - 建议使用合适的缓冲区大小
   - 确保视频编码格式兼容

## 调试信息

前端会在控制台输出详细的调试信息，包括：
- 连接状态变化
- 请求参数详情
- 接收到的消息内容
- 错误和警告信息
- 视频数据处理状态

这些信息可以帮助排查问题，建议后端也添加相应的日志记录。 