import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class PipeFlowSimulator:
    """
    A class for simulating and visualizing pipe flow using the Darcy-Weisbach equation.
    """
    def __init__(self, master):
        """
        Initializes the Pipe Flow Simulator GUI.

        Parameters:
        master (tk.Tk): The root window of the Tkinter application.
        """
        self.master = master
        self.master.title("Pipe Flow Simulator")

        # --- Input Frame ---
        self.input_frame = ttk.Frame(master, padding="10")
        self.input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Labels and Entry widgets for input parameters
        ttk.Label(self.input_frame, text="Pipe Length (m):").grid(row=0, column=0, sticky=tk.W)
        self.length_entry = ttk.Entry(self.input_frame)
        self.length_entry.grid(row=0, column=1, sticky=tk.W)
        self.length_entry.insert(0, "100")  # Default value

        ttk.Label(self.input_frame, text="Pipe Diameter (m):").grid(row=1, column=0, sticky=tk.W)
        self.diameter_entry = ttk.Entry(self.input_frame)
        self.diameter_entry.grid(row=1, column=1, sticky=tk.W)
        self.diameter_entry.insert(0, "0.1")  # Default value

        ttk.Label(self.input_frame, text="Flow Rate (m3/s):").grid(row=2, column=0, sticky=tk.W)
        self.flow_rate_entry = ttk.Entry(self.input_frame)
        self.flow_rate_entry.grid(row=2, column=1, sticky=tk.W)
        self.flow_rate_entry.insert(0, "0.01")  # Default value

        ttk.Label(self.input_frame, text="Kinematic Viscosity (m2/s):").grid(row=3, column=0, sticky=tk.W)
        self.viscosity_entry = ttk.Entry(self.input_frame)
        self.viscosity_entry.grid(row=3, column=1, sticky=tk.W)
        self.viscosity_entry.insert(0, "1e-6")  # Default value

        ttk.Label(self.input_frame, text="Roughness (m):").grid(row=4, column=0, sticky=tk.W)
        self.roughness_entry = ttk.Entry(self.input_frame)
        self.roughness_entry.grid(row=4, column=1, sticky=tk.W)
        self.roughness_entry.insert(0, "1.5e-6")  # Default value

        # Calculate Button
        self.calculate_button = ttk.Button(self.input_frame, text="Calculate", command=self.calculate_and_plot)
        self.calculate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # --- Results Frame (Initially Empty) ---
        self.results_frame = ttk.Frame(master, padding="10")
        self.results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_frame.columnconfigure(0, weight=1)  # Make the column resizable

        self.plot_label = ttk.Label(self.results_frame, text="Results will be displayed here.")
        self.plot_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # --- Status Bar ---
        self.status_label = ttk.Label(master, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # --- Menu ---
        self.create_menu()

        # --- Error Handling ---
        self.error_occurred = False

    def create_menu(self):
        """
        Creates the application's menu bar.
        """
        menu_bar = tk.Menu(self.master)
        self.master.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.master.destroy)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        menu_bar.add_cascade(label="Help", menu=help_menu)

    def show_about_dialog(self):
        """
        Displays an "About" dialog box.
        """
        messagebox.showinfo("About Pipe Flow Simulator",
                            "This application simulates pipe flow using the Darcy-Weisbach equation.\n"
                            "Version 1.0")

    def calculate_and_plot(self):
        """
        Calculates the head loss and plots the pressure distribution along the pipe.
        Handles errors in input data.
        """
        self.status_label.config(text="Calculating...")
        self.error_occurred = False  # Reset error flag

        try:
            # Get input values from Entry widgets
            length = float(self.length_entry.get())
            diameter = float(self.diameter_entry.get())
            flow_rate = float(self.flow_rate_entry.get())
            viscosity = float(self.viscosity_entry.get())
            roughness = float(self.roughness_entry.get())

            # Perform the calculation
            head_loss, friction_factor, velocity, reynolds_number = self.calculate_pipe_flow(length, diameter, flow_rate, viscosity, roughness)

            # Generate and display the plot
            self.plot_pressure_distribution(length, head_loss, diameter)
            self.status_label.config(text="Calculation complete. Results displayed.")

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            self.status_label.config(text="Error: Invalid input")
            self.error_occurred = True
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.status_label.config(text="Error: " + str(e))
            self.error_occurred = True

        if not self.error_occurred:
            self.status_label.config(text="Ready") # change status

    def calculate_pipe_flow(self, length, diameter, flow_rate, viscosity, roughness):
        """
        Calculates head loss using the Darcy-Weisbach equation.

        Parameters:
        length (float): Length of the pipe (m).
        diameter (float): Diameter of the pipe (m).
        flow_rate (float): Flow rate (m3/s).
        viscosity (float): Kinematic viscosity of the fluid (m2/s).
        roughness (float): Roughness of the pipe wall (m).

        Returns:
        tuple: (head_loss, friction_factor, velocity, reynolds_number)
               Returns None if there is an error
        """

        # Calculate cross-sectional area and velocity
        area = np.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        #print(f"Velocity: {velocity}")

        # Calculate Reynolds number
        reynolds_number = (velocity * diameter) / viscosity
        #print(f"Reynolds Number: {reynolds_number}")

        # Determine flow regime and calculate friction factor
        if reynolds_number < 2000:
            # Laminar flow
            friction_factor = 64 / reynolds_number
        elif reynolds_number > 4000:
            # Turbulent flow - use Colebrook-White equation (iterative solution)
            def colebrook_equation(f):
                return 1 / np.sqrt(f) + 2 * np.log10(roughness / (3.7 * diameter) + 2.51 / (reynolds_number * np.sqrt(f)))

            # Initial guess for friction factor
            friction_factor = 0.02
            for _ in range(10):  # Iterate to converge
                friction_factor = 1 / colebrook_equation(friction_factor)**2
        else:
            # Transitional flow -  A simple interpolation (can be improved)
            f_laminar = 64 / reynolds_number
            def colebrook_equation(f):
                return 1 / np.sqrt(f) + 2 * np.log10(roughness / (3.7 * diameter) + 2.51 / (reynolds_number * np.sqrt(f)))
            f_turbulent = 1 / colebrook_equation(0.02)**2 #use 0.02 as initial guess
            friction_factor = f_laminar + (f_turbulent - f_laminar) * (reynolds_number - 2000) / 2000

        # Calculate head loss using Darcy-Weisbach equation
        head_loss = friction_factor * (length / diameter) * (velocity ** 2 / (2 * 9.81))
        #print(f"Friction Factor: {friction_factor}")
        #print(f"Head Loss: {head_loss}")
        return head_loss, friction_factor, velocity, reynolds_number

    def plot_pressure_distribution(self, length, head_loss, diameter):
        """
        Plots the pressure distribution along the pipe.

        Parameters:
        length (float): Length of the pipe (m).
        head_loss (float): Total head loss in the pipe (m).
        diameter (float): Diameter of the pipe
        """
        # Generate points along the pipe
        num_points = 100
        x = np.linspace(0, length, num_points)
        # Assume initial pressure at the start of the pipe (x=0)
        initial_pressure = 100000  # Pa (adjust as needed)
        # Calculate pressure at each point along the pipe
        pressure_drop_per_meter = (head_loss * 9.81 * 1000) / length  # Pa/m  (rho = 1000 kg/m3)
        pressure = initial_pressure - pressure_drop_per_meter * x

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, pressure)
        ax.set_xlabel("Distance along pipe (m)")
        ax.set_ylabel("Pressure (Pa)")
        ax.set_title("Pressure Distribution along Pipe")
        ax.grid(True)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        # Use grid instead of pack
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Make sure the plotの位置 expands with the frame.
        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.plot_label.destroy() # remove the text label.

if __name__ == "__main__":
    root = tk.Tk()
    app = PipeFlowSimulator(root)
    root.mainloop()
