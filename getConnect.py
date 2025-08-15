import re

with open('subtitles.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 匹配时间戳格式的正则表达式
pattern = re.compile(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]")

# 移除时间戳但保留换行
clean_lines = [pattern.sub('', line).lstrip() for line in lines]

with open('cleaned.txt', 'w', encoding='utf-8') as f:
    f.writelines(clean_lines)