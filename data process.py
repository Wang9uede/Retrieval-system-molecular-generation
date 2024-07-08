import csv
import json

# 定义输出文件名
output_json = 'output.json'
output_csv = 'output.csv'

# 打开输出文件
with open(output_json, 'w') as json_file, open(output_csv, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    json_file.write('[\n')  # 开始JSON数组

    # 逐行读取文件
    with open('/Users/wangjuede/Downloads/molxpt/molxpt_code/molxpt_subset_data/train.mix.subset', 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # 按句号分割句子
            sentences = line.strip().split('.')

            # 写入CSV文件
            for sentence in sentences:
                if sentence:  # 确保不为空
                    csv_writer.writerow([sentence])

            # 写入JSON文件
            # 为了可读性，这里限制了每行输出的句子数量
            json_file.write(',\n'.join([f'  "{sentence}"' for sentence in sentences]))
            if line_number % 1000 == 0:  # 每1000句换行
                json_file.write(',\n')

    json_file.write('\n]\n')  # 结束JSON数组
