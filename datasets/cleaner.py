


with open('result.txt', 'r') as in_file:
    with open('result2.txt','w') as out_file:
        for line in in_file:
            line = line.strip()
            if line : out_file.write(f'{line} [END]\n')
