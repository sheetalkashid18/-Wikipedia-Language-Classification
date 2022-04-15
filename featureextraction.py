import pandas as pd
fname = "rawdata"
f = open(fname, "r", encoding='utf-8')

data = []
for line in f:
    row = []
    div = line.split("|")
    lang = div[0]
    countOfkj = 0
    numberofwords = 0
    totallength = 0
    numberofdoubleletters = 0
    istrue1 = False
    istrue2 = False

    for i in div[1].split():
        #print(i)
        numberofwords = numberofwords + 1
        prev = None
        for j in i:
            #print(j)
            totallength = totallength + 1

            if j == 'k' or j == 'j':
                countOfkj = countOfkj + 1

            if prev is not None:
                if j == prev:
                    numberofdoubleletters = numberofdoubleletters + 1
            prev = j

        ilow = i.lower()
        if not istrue1:
            if ilow == 'she' or ilow == 'the' or ilow == 'a' or ilow == 'an' or ilow == 'and' or ilow == 'you' or ilow == 'but':
                print(i)
                #row.append('Yes')
                istrue1 = True


        if not istrue2:
            if ilow == 'het' or ilow == 'de' or ilow == 'een' or ilow == 'en' or ilow == 'de' or ilow =='eng' or ilow =='maar':
                #row.append('Yes')
                istrue2 = True


    if istrue1:
        row.append('Yes')
    else:
        row.append('No')

    if istrue2:
        row.append('Yes')
    else:
        row.append('No')


    if totallength/numberofwords > 5:
        row.append('Yes')
    else:
        row.append('No')


    if countOfkj > 3:
        row.append('Yes')
    else:
        row.append('No')


    if numberofdoubleletters > 2:
        row.append('Yes')
    else:
        row.append('No')

    if 'q' in div[1].split():
        row.append('Yes')
    else:
        row.append('No')


    row.append(lang)

    data.append(row)


data = pd.DataFrame(data)


print(data)
