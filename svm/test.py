import test_data as data

x = data.TestData(40, True)
x.generate_data()

print(f"Inputs: \n {x.inputs}")
print(f"Targets: \n {x.targets}")

print(data.targets)
