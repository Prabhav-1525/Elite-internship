from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, lpSum

def get_input(prompt, default=None, cast_type=float):
    while True:
        try:
            value = input(f"{prompt} [{default if default is not None else 'required'}]: ")
            if value == '' and default is not None:
                return default
            return cast_type(value)
        except ValueError:
            print("Please enter a valid number!")

print("Welcome to the Product Mix Optimization Tool!")
print("You will be asked to enter product details and resource constraints.")

# User inputs for product names and parameters
product_names = []
products = []
profit = {}
labor = {}
material = {}

num_products = get_input("Enter the number of products", 2, int)
for i in range(1, num_products+1):
    name = input(f"Enter name for product {i}: ")
    product_names.append(name)
    profit[name] = get_input(f"Profit per unit of {name} (₹)", 40)
    labor[name] = get_input(f"Labor hours per unit of {name}", 2)
    material[name] = get_input(f"Material (kg) per unit of {name}", 1)

max_labor = get_input("Maximum labor available (hours)", 100)
max_material = get_input("Maximum material available (kg)", 60)

# Model
model = LpProblem(name="Product_Mix_Optimization", sense=LpMaximize)
vars = {name: LpVariable(name=name, lowBound=0) for name in product_names}

# Objective
model += lpSum(profit[name] * var for name, var in vars.items()), "Total_Profit"

# Constraints
model += lpSum(labor[name] * var for name, var in vars.items()) <= max_labor, "Labor_Constraint"
model += lpSum(material[name] * var for name, var in vars.items()) <= max_material, "Material_Constraint"

# Solve
model.solve()

# Results
print("\n--- Results ---")
print(f"Status: {LpStatus[model.status]}")
print(f"Objective (Total Profit): ₹{model.objective.value():.2f}")
print("Optimal production quantities:")
for name, var in vars.items():
    print(f"{name}: {var.value():.2f} units")

# Slack
print("\nResource Utilization:")
for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.slack:.2f} (unused resource)")