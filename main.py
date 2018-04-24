print("Started")


def testMeth(func, x):
    func(x)


for i in range(10):
    testMeth(lambda x: print(x), i)


MAX = 2
print("Finished")
