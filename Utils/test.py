import sys
import time 
import random 
globalVarMin =float("inf")
globalVarMax = float("-inf")

def ResetGlobals():
    global globalVarMin 
    global globalVarMax
    globalVarMin =float("inf")
    globalVarMax = float("-inf")
def FileRead(path):
    try:
        file = open(path, "r")
    except FileNotFoundError:
        print("File:" + path + " not found")
        exit()
    except ValueError:
        print("File format is invalid")
        exit()
    text = file.readlines()
    print(text)
    formated_data =[[float(number) for number in   line.strip().split()] for line in text]
    size  = formated_data.pop(0) 
    return ( size ,formated_data)


def MinFunc(lst,result):
    new_set = set(range(len(lst))) - set(result)
    min = float("inf")
    for element in new_set:
        if min > lst[element] and lst[element] != 0 :
            min = lst[element]
                
    return lst.index(min)



def BackTrack(startLocation,matrix ,size,result ):
    global globalVarMin
    global globalVarMax
    result.append(startLocation)
    if len(result) == size:           
        weight = 0
        for i in range(len(result)-1):
            weight += (matrix[result[i]][result[i+1]])
        if(globalVarMin > weight):
            globalVarMin = weight
        if globalVarMax < weight:
            globalVarMax = weight
      #  print(f"{result}; Weight:{weight}")

    else:
        for next_index in range(len(matrix[startLocation])):
            if next_index not in  result and next_index != startLocation:
                BackTrack(next_index,matrix,size,result)
    result.pop()       
                
        
def Optimized(startLocation,matrix ,size ):
    global globalVarMin
    global globalVarMax
    result = []
    result.append(startLocation)

    for node in matrix:
        if(len(result) == size):
            break
        min_index = MinFunc(matrix[startLocation],result)
        result.append(min_index)
       # print(result)
    weight = 0
    for i in range(len(result)-1):
        weight += (matrix[result[i]][result[i+1]])
    if(globalVarMin > weight):
        globalVarMin = weight
    if globalVarMax < weight:
        globalVarMax = weight
    print(f"{result}; Weight:{weight}")    
    return result

def RandomTraverse(startLocation,matrix ,size ):
    global globalVarMin
    global globalVarMax
    result = []
    result.append(startLocation)
    index =  0
    for node in matrix:
        if(len(result) == size):
            break
        while(True):
            index = random.randint(0,size-1)
            if index not in result:
                break
        
        result.append(index)
       # print(result)
    weight = 0
    for i in range(len(result)-1):
        weight += (matrix[result[i]][result[i+1]])
    if(globalVarMin > weight):
        globalVarMin = weight
    if globalVarMax < weight:
        globalVarMax = weight
    print(f"{result}; Weight:{weight}")    
    return result
    









if __name__ == "__main__":
   
    size,matrix = FileRead("input.txt")
    result = []
    size =  int(size[0])
    '''
    print("\nBackTracking")
    for i in range(size):
    
        start = time.time()
        BackTrack(i,matrix,size,result)
        stop = time.time()
        print(f"{i}:Execution time : {stop-start}" )
    print("GlobalVarMin = " , globalVarMin)
    print("GlobalVarMax = " , globalVarMax)
    '''

    ResetGlobals()
    print("\nOptimized")
    for i in range(size):
        start = time.time()
        Optimized(i,matrix ,size )
        stop = time.time()
        print(f"{i}:Execution time : {stop-start}" )
    print("GlobalVarMin = " , globalVarMin)
    print("GlobalVarMax = " , globalVarMax)
    ResetGlobals()
    print("\nRandomTraverse")
    for i in range(size):
        start = time.time()
        RandomTraverse(i,matrix ,size )
        stop = time.time()
        print(f"{i}:Execution time : {stop-start}" )
    print("GlobalVarMin = " , globalVarMin)
    print("GlobalVarMax = " , globalVarMax)

