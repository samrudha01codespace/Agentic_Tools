import sympy as sp
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


# --- 1. Symbolic Math (SymPy) ---
def solve_symbolic():
    x = sp.Symbol('x')
    equation = sp.Eq(x ** 2 + 2 * x - 3, 0)
    solutions = sp.solve(equation, x)
    print(f"Symbolic solution of {equation}: {solutions}")


# --- 2. Thermodynamic Properties (CoolProp) ---
def get_water_properties(temp_c):
    T = temp_c + 273.15  # Convert °C to K
    P = PropsSI('P', 'T', T, 'Q', 0, 'Water')  # Saturation pressure
    H = PropsSI('H', 'T', T, 'Q', 0, 'Water')  # Enthalpy
    print(f"At {temp_c}°C: Pressure = {P / 1e5:.2f} bar, Enthalpy = {H / 1000:.2f} kJ/kg")


# --- 3. Plotting (Matplotlib) ---
def plot_temperature_vs_pressure():
    temps = list(range(0, 201, 10))
    pressures = [PropsSI('P', 'T', t + 273.15, 'Q', 0, 'Water') / 1e5 for t in temps]
    plt.plot(temps, pressures, marker='o')
    plt.title("Saturation Pressure vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure (bar)")
    plt.grid(True)
    plt.show()


# --- 4. Placeholder: Mechanical Simulation (FreeCAD / Blender) ---
def simulate_mechanical():
    print("Launching FreeCAD simulation... (placeholder)")
    # Could call FreeCAD scripts here


# --- 5. Placeholder: Nuclear Simulation (OpenMC) ---
def run_openmc_simulation():
    print("Running OpenMC neutron simulation... (placeholder)")
    # Normally you'd call: openmc.run()


# --- 6. Placeholder: Circuit Simulation (LTspice) ---
def run_circuit_simulation():
    print("Simulating circuit in LTspice... (placeholder)")
    # Could launch LTspice scripts or parse .net files


# --- Main Menu ---
def main():
    print("🤖 Engineering Jarvis")
    print("1. Solve symbolic equation")
    print("2. Get water properties")
    print("3. Plot saturation pressure curve")
    print("4. Simulate mechanical (FreeCAD)")
    print("5. Run nuclear sim (OpenMC)")
    print("6. Run circuit sim (LTspice)")

    choice = input("Choose an option: ")

    if choice == '1':
        solve_symbolic()
    elif choice == '2':
        temp = float(input("Enter temperature in °C: "))
        get_water_properties(temp)
    elif choice == '3':
        plot_temperature_vs_pressure()
    elif choice == '4':
        simulate_mechanical()
    elif choice == '5':
        run_openmc_simulation()
    elif choice == '6':
        run_circuit_simulation()
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
