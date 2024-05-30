from PIL import Image

# 打开 GIF 文件
gif_path = "D:\桌面\GD/特征映射.gif"
gif = Image.open(gif_path)

# 获取 GIF 中的最后一帧
last_frame = gif.seek(gif.n_frames - 1)
last_frame_image = gif.copy()

# 保存最后一帧为静态图像
last_frame_image.save("last_frame.png")
