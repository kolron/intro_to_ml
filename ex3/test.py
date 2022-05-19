



filename = r'D:\school\machine learning\ex3\four_circle.txt'
out_f = r'D:\school\machine learning\ex3\four_circle.csv'

data = []
with open(filename) as f:
    for line in f:
        new_line = line.split(" ")
        str = ''
        for w in new_line:
            str += f'{w},'
        str = str[:len(str)-1]
        data += [str]
    print(data)

with open(out_f,'a+') as f:
    for line in data:
        f.write(line)
