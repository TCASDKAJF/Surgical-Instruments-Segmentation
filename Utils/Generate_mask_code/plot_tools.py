import os
import random
from PIL import Image, ImageDraw, ImageFont

# 工具类别及其数量
tools_data = {
    'freer_elevator': 1810,
    'spatula_dissector': 1039,
    'kerrisons': 6420,
    'suction': 8670,
    'retractable_knife': 1889,
    'dural_scissors': 947,
    'pituitary_rongeurs': 2154,
    'surgiflo': 495,
    'bipolar_forceps': 464,
    'blakesley': 115,
    'ring_curette': 414,
    'cottle': 342,
    'doppler': 127,
    'drill': 287,
    'cup_forceps': 220,
    'stealth_pointer': 556
}

base_folder = "G:\\baseline\\Swin\\polyp-seg\\tool_comparison"
selected_images = []

for tool, count in tools_data.items():
    tool_folder = os.path.join(base_folder, tool)
    image_name = random.choice(os.listdir(tool_folder))
    selected_images.append((Image.open(os.path.join(tool_folder, image_name)), tool, count))

img_width, img_height = selected_images[0][0].size
font = ImageFont.truetype("arial.ttf", 60)  # 调整字体大小

# 计算文字的实际高度
draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
_, _, _, text_height_actual = draw.textbbox((0, 0), "测试", font=font)

# 设定总的图片大小
total_width = 8 * img_width
# 图片高度乘以2（因为有两行图片） + 文本的高度乘以2（因为每张图片下有文字）
total_height = 2 * img_height + 2* (text_height_actual + 10)  # 10为间隙

result_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
draw = ImageDraw.Draw(result_image)

for index, (img, tool, count) in enumerate(selected_images):
    col = index % 8
    x_offset = col * img_width

    row = index // 8
    y_offset = row * (img_height + text_height_actual + 10)  # 图片高度加上文本高度和间隙

    # 放置图片
    result_image.paste(img, (x_offset, y_offset))

    # 写入文本
    title_text = f"{tool} ({count})"
    _, _, text_width, _ = draw.textbbox((0, 0), title_text, font=font)
    text_position = (x_offset + (img_width - text_width) / 2, y_offset + img_height + 5)
    draw.text(text_position, title_text, fill="black", font=font)

result_image.show()

input_text = input("如果图片无误，请输入'yes'保存，否则输入'no'退出：")
if input_text.lower() == 'yes':
    result_image.save("G:\\baseline\\Swin\\polyp-seg\\combined_image.jpg")
    print("图片已保存！")
else:
    print("操作已取消。")
# import os
# import random
# from PIL import Image, ImageDraw, ImageFont
#
# # 工具类别及其数量
# tools_data = {
#     'freer_elevator': 1810,
#     'spatula_dissector': 1039,
#     'kerrisons': 6420,
#     'suction': 8670,
#     'retractable_knife': 1889,
#     'dural_scissors': 947,
#     'pituitary_rongeurs': 2154,
#     'surgiflo': 495,
#     'bipolar_forceps': 464,
#     'blakesley': 115,
#     'ring_curette': 414,
#     'cottle': 342,
#     'doppler': 127,
#     'drill': 287,
#     'cup_forceps': 220,
#     'stealth_pointer': 556
# }
#
# base_folder = "G:\\baseline\\Swin\\polyp-seg\\tool_comparison"
# output_folder = "G:\\baseline\\Swin\\polyp-seg\\output_images"
#
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# font = ImageFont.truetype("arial.ttf", 60)
# draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
# _, _, _, text_height_actual = draw.textbbox((0, 0), "测试", font=font)
#
# for i in range(100):
#     selected_images = []
#     for tool, count in tools_data.items():
#         tool_folder = os.path.join(base_folder, tool)
#         image_name = random.choice(os.listdir(tool_folder))
#         selected_images.append((Image.open(os.path.join(tool_folder, image_name)), tool, count))
#
#     img_width, img_height = selected_images[0][0].size
#     total_width = 8 * img_width
#     total_height = 2 * img_height + 2 * (text_height_actual + 10)
#     result_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
#     draw = ImageDraw.Draw(result_image)
#
#     for index, (img, tool, count) in enumerate(selected_images):
#         col = index % 8
#         x_offset = col * img_width
#         row = index // 8
#         y_offset = row * (img_height + text_height_actual + 10)
#         result_image.paste(img, (x_offset, y_offset))
#         title_text = f"{tool} ({count})"
#         _, _, text_width, _ = draw.textbbox((0, 0), title_text, font=font)
#         text_position = (x_offset + (img_width - text_width) / 2, y_offset + img_height + 5)
#         draw.text(text_position, title_text, fill="black", font=font)
#
#     result_image.save(os.path.join(output_folder, f"combined_image_{i+1}.jpg"))
#
# print("100张图片已生成并保存在指定目录！")




