def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    if len(a[0]) != len(b) :
        return -1
    else : 
        z = []
        for i in range(len(b)) : 
            temp = []
            for j in range(len(b)):
                 temp.append(b[j][i])
            z.append(temp)

        f = []
        for i in a:
            temp = []
            for j in z:
                s = 0
                for c in range(len(i)):
                    s = s + i[c] * j[c]

                temp.append(s)
            
            f.append(temp)

            
	return f