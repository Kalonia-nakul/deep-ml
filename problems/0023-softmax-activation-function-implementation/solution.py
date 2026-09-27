import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here

    # softmax = []
    # x = 0 
    # for i in range(len(scores)):
    #     x = x + math.exp(scores[i])

    # for i in range(len(scores)):
    #     y = math.exp(scores[i])
    #     softmax.append( y / x )
    
    # return softmax 

    max_value = max(scores)

    exp_values = [math.exp(x - max_value) for x in scores]

    total = sum(exp_values)

    return [x / total for x in exp_values]