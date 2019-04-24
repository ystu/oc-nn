
myDict = {}
myDict['c'] = 1
myDict['b'] = 3
myDict['a'] = 2



print(myDict)

otherDict = sorted(myDict, key=myDict.get)

print(myDict)
print(otherDict)

print(type(myDict.items()))

# myList = myDict.items()

myList = [(key, value) for key, value in myDict.items()]
print(type(myList))
print(type(myList[0]))
print(myList[0][0])

