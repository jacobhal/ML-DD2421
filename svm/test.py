import test_data as data

x = data.TestData(40, True)
x.generate_data()

print(f"Inputs: \n {x.inputs}")
print(f"Targets: \n {x.targets}")

a = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

res = [x[0] for x in a]

print(res)
