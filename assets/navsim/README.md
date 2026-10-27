# NAVSIM 清单

本机目录已包含 `train.jsonl`、`val.jsonl`、`norm_stats.json`、`audit.json`，Git 默认忽略这些实际数据文件。完整资产交接包会包含它们。

每行格式：

```json
{
  "token": "scene-token",
  "group": "original-recording",
  "images": {
    "cam_front_center": "sensor_blobs/trainval/log/CAM_F0/image.jpg",
    "cam_front_left": "sensor_blobs/trainval/log/CAM_L0/image.jpg",
    "cam_front_right": "sensor_blobs/trainval/log/CAM_R0/image.jpg"
  },
  "state": ["speed", "vx", "vy", "ax", "ay"],
  "prompt": "Drive safely. Recorded navigation command: 0 1 0 0",
  "actions": [["x", "y", "heading"]]
}
```

示意中的 state/actions 字符串应为真实数值；actions 必须恰好有 8 行，表示当前自车坐标系的未来绝对位姿，采样间隔 0.5 秒，heading 为弧度。相机顺序严格为 F0、L0、R0。后视相机即使出现在旧清单中也不会读取。

当前导入：训练 30,266，验证 7,569；保持旧项目的 recording-disjoint 自定义划分。新机器修改 `data.root` 即可重定位图像。原图没有包含在项目中。统计仅从训练集计算，不要用验证集或 navtest 重算。

生成命令与拆分说明见 [使用说明](../../docs/使用说明.md)。
