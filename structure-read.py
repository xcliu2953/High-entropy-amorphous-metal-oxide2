import os
import openpyxl

# 定义字母和对应的数值的字典
letter_values = {
    'G': 0.732592,
    'J': 0.496159,
    'M': 0.265455,
    'E': 0.224832,
    'A': 0.225148,
    'L': 0.247287,
    'O': 0.993281,
    'N': 0.761506,
    'B': 0.706126,
    'H': 0.480956,
    'C': 0.955664,
    'I': 0.007142,
    'F': 0.96499,
    'P': 0.488512,
    'D': 0.450336,
    'K': 0.753633
}

# 创建一个Excel工作簿
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = "Results"

# 要遍历的文件名列表
file_names1 = [
    "POSCAR1.txt",
    "POSCAR111.txt",
    "POSCAR122.txt",
    "POSCAR2.txt",
    "POSCAR211.txt",
    "POSCAR222.txt",
    "POSCAR3.txt",
    "POSCAR311.txt",
    "POSCAR322.txt",
    "POSCAR4.txt",
    "POSCAR411.txt",
    "POSCAR422.txt",
    "POSCAR5.txt",
    "POSCAR511.txt",
    "POSCAR522.txt",
    "POSCAR6.txt",
    "POSCAR611.txt",
    "POSCAR622.txt",
    "POSCAR7.txt",
    "POSCAR711.txt",
    "POSCAR722.txt",
    "POSCAR8.txt",
    "POSCAR811.txt",
    "POSCAR822.txt",
    "POSCAR9.txt",
    "POSCAR911.txt",
    "POSCAR922.txt",
    "POSCAR10.txt",
    "POSCAR1011.txt",
    "POSCAR1022.txt",
    "POSCAR11.txt",
    "POSCAR1111.txt",
    "POSCAR1122.txt",
    "POSCAR12.txt",
    "POSCAR1211.txt",
    "POSCAR1222.txt",
    # 这里继续添加其他文件名
]
file_names2 = [
    "POSCAR1.txt",
    "POSCAR133.txt",
    "POSCAR144.txt",
    "POSCAR2.txt",
    "POSCAR233.txt",
    "POSCAR244.txt",
    "POSCAR3.txt",
    "POSCAR333.txt",
    "POSCAR344.txt",
    "POSCAR4.txt",
    "POSCAR433.txt",
    "POSCAR444.txt",
    "POSCAR5.txt",
    "POSCAR533.txt",
    "POSCAR544.txt",
    "POSCAR6.txt",
    "POSCAR633.txt",
    "POSCAR644.txt",
    "POSCAR7.txt",
    "POSCAR733.txt",
    "POSCAR744.txt",
    "POSCAR8.txt",
    "POSCAR833.txt",
    "POSCAR844.txt",
    "POSCAR9.txt",
    "POSCAR933.txt",
    "POSCAR944.txt",
    "POSCAR10.txt",
    "POSCAR1033.txt",
    "POSCAR1044.txt",
    "POSCAR11.txt",
    "POSCAR1133.txt",
    "POSCAR1144.txt",
    "POSCAR12.txt",
    "POSCAR1233.txt",
    "POSCAR1244.txt",
    # 这里继续添加其他文件名
]
file_names3 = [
    "POSCAR1.txt",
    "POSCAR133.txt",
    "POSCAR144.txt",
    "POSCAR2.txt",
    "POSCAR233.txt",
    "POSCAR244.txt",
    "POSCAR3.txt",
    "POSCAR333.txt",
    "POSCAR344.txt",
    "POSCAR4.txt",
    "POSCAR433.txt",
    "POSCAR444.txt",
    "POSCAR5.txt",
    "POSCAR533.txt",
    "POSCAR544.txt",
    "POSCAR6.txt",
    "POSCAR633.txt",
    "POSCAR644.txt",
    "POSCAR7.txt",
    "POSCAR733.txt",
    "POSCAR744.txt",
    "POSCAR8.txt",
    "POSCAR833.txt",
    "POSCAR844.txt",
    "POSCAR9.txt",
    "POSCAR933.txt",
    "POSCAR944.txt",
    "POSCAR10.txt",
    "POSCAR1033.txt",
    "POSCAR1044.txt",
    "POSCAR11.txt",
    "POSCAR1133.txt",
    "POSCAR1144.txt",
    "POSCAR12.txt",
    "POSCAR1244.txt",
    # 这里继续添加其他文件名
]
# 获取文件夹中的文件并遍历
for folder_name in range(1, 2):  # 遍历4-2到4-12文件夹
    folder_path = f"4-{folder_name}"  # 设置文件夹路径

    # 遍历文件名列表中的文件
    for file_name in file_names1:
        file_path = os.path.join(folder_path, file_name)

        # 打开文件并读取指定行的第一个数
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 初始化一个列表用于存储匹配的字母
        matched_letters = []

        # 检查是否有足够的行数
        if len(lines) >= 57:
            for i in range(42, 58):  # 遍历第42到57行
                line = lines[i - 1]  # Python中索引从0开始，所以需要减1
                numbers = line.split()  # 将行按空格分割成单词
                if len(numbers) >= 1:
                    first_number = float(numbers[0])

                    # 检查第一个数是否在字母和对应数值的字典中
                    for letter, value in letter_values.items():
                        if abs(first_number - value) < 1e-6:  # 使用一个小的容差来比较浮点数
                            matched_letters.append(letter)
                            break

        # 打印匹配的字母
        print(f"{folder_path}/{file_name}: {', '.join(matched_letters)}")

        # 将匹配的字母按照行的顺序写入Excel
        sheet.append([f"{folder_path}/{file_name}"] + matched_letters)
# 获取文件夹中的文件并遍历
for folder_name in range(2, 12):  # 遍历4-2到4-12文件夹
    folder_path = f"4-{folder_name}"  # 设置文件夹路径

    # 遍历文件名列表中的文件
    for file_name in file_names2:
        file_path = os.path.join(folder_path, file_name)

        # 打开文件并读取指定行的第一个数
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 初始化一个列表用于存储匹配的字母
        matched_letters = []

        # 检查是否有足够的行数
        if len(lines) >= 57:
            for i in range(42, 58):  # 遍历第42到57行
                line = lines[i - 1]  # Python中索引从0开始，所以需要减1
                numbers = line.split()  # 将行按空格分割成单词
                if len(numbers) >= 1:
                    first_number = float(numbers[0])

                    # 检查第一个数是否在字母和对应数值的字典中
                    for letter, value in letter_values.items():
                        if abs(first_number - value) < 1e-6:  # 使用一个小的容差来比较浮点数
                            matched_letters.append(letter)
                            break

        # 打印匹配的字母
        print(f"{folder_path}/{file_name}: {', '.join(matched_letters)}")

        # 将匹配的字母按照行的顺序写入Excel
        sheet.append([f"{folder_path}/{file_name}"] + matched_letters)
# 获取文件夹中的文件并遍历
for folder_name in range(12, 13):  # 遍历4-2到4-12文件夹
    folder_path = f"4-{folder_name}"  # 设置文件夹路径

    # 遍历文件名列表中的文件
    for file_name in file_names3:
        file_path = os.path.join(folder_path, file_name)

        # 打开文件并读取指定行的第一个数
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 初始化一个列表用于存储匹配的字母
        matched_letters = []

        # 检查是否有足够的行数
        if len(lines) >= 57:
            for i in range(42, 58):  # 遍历第42到57行
                line = lines[i - 1]  # Python中索引从0开始，所以需要减1
                numbers = line.split()  # 将行按空格分割成单词
                if len(numbers) >= 1:
                    first_number = float(numbers[0])

                    # 检查第一个数是否在字母和对应数值的字典中
                    for letter, value in letter_values.items():
                        if abs(first_number - value) < 1e-6:  # 使用一个小的容差来比较浮点数
                            matched_letters.append(letter)
                            break

        # 打印匹配的字母
        print(f"{folder_path}/{file_name}: {', '.join(matched_letters)}")

        # 将匹配的字母按照行的顺序写入Excel
        sheet.append([f"{folder_path}/{file_name}"] + matched_letters)
# 保存Excel文件
workbook.save("matched_letters.xlsx")

# 关闭工作簿
workbook.close()
