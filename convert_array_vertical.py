
#讀取輸入的txt檔案，將內容轉換為每行一個數字的格式輸出到新檔案

def convert_txt_vertical(input_file_path, output_file_path):

    with open(input_file_path, 'r') as f:
        content = f.read()

    for char in ['(', ')', '[', ']', '{', '}', ',', ';']:
        content = content.replace(char, ' ')
        
    numbers = content.split()

    with open(output_file_path, 'w') as f:
        for num in numbers:
            try:
                num_value = float(num)
                if num_value.is_integer():
                    f.write(f"{int(num_value)}\n")
                else:
                    f.write(f"{num_value}\n")
            except ValueError:
                continue

convert_txt_vertical("./input/input.txt", "./input/input_.txt")
