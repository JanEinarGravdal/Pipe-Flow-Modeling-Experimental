# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class InteractiveFluidSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Non-Newtonian Fluid Simulator")
        
        # Simulation control
        self.sim_running = False
        self.P_outlet = 101325.0  # Initial outlet pressure [Pa]
        self.lock = threading.Lock()
        
        # Data storage
        self.time_history = []
        self.P_inlet_history = []
        self.Q_in_history = []
        self.Q_out_history = []
        self.velocity_profiles = []
        self.pressure_profiles = []
        
        # GUI setup
        self.create_controls()
        self.create_plots()
        self.create_parameter_inputs()
        
        # Simulation thread
        self.sim_thread = None

    def create_controls(self):
        """Create control widgets"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Outlet pressure control
        ttk.Label(control_frame, text="Outlet Pressure (bar):").pack(pady=5)
        self.p_outlet_slider = ttk.Scale(
            control_frame, from_=1.0, to=3.0, orient=tk.HORIZONTAL,
            command=lambda v: self.update_outlet_pressure(float(v)*1e5)
        )
        self.p_outlet_slider.set(self.P_outlet/1e5)
        self.p_outlet_slider.pack(pady=5)

        # Control buttons
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(pady=10)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

    def create_parameter_inputs(self):
        """Create input fields for simulation parameters"""
        param_frame = ttk.Frame(self.root)
        param_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.entries = {}
        params = [
            ("Pipe Length (m)", 2500.0),
            ("Pipe Diameter (m)", 0.05),
            ("Bulk Modulus (GPa)", 1.0),
            ("Initial Viscosity (Pa·s)", 0.035),
            ("Consistency Index (K)", 0.1),
            ("Flow Index (n)", 0.8)
        ]

        for i, (label, default) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            entry = ttk.Entry(param_frame)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, pady=2)
            self.entries[label] = entry

    def create_plots(self):
        """Create matplotlib plots embedded in Tkinter"""
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def update_outlet_pressure(self, p_bar):
        """Update outlet pressure with thread-safe operation"""
        with self.lock:
            self.P_outlet = p_bar * 1e5

    def start_simulation(self):
        """Start the simulation thread"""
        if not self.sim_running:
            self.sim_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.sim_thread = threading.Thread(target=self.run_simulation)
            self.sim_thread.start()

    def stop_simulation(self):
        """Stop the simulation gracefully"""
        self.sim_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_plots(self):
        """Update all plots with current data"""
        try:
            # Clear previous plots
            for ax in self.axs.flatten():
                ax.clear()

            # Plot inlet pressure
            self.axs[0,0].plot(self.time_history, self.P_inlet_history)
            self.axs[0,0].set_title("Inlet Pressure")
            self.axs[0,0].set_xlabel("Time [s]")
            self.axs[0,0].set_ylabel("Pressure [bar]")

            # Plot flow rates
            self.axs[0,1].plot(self.time_history, self.Q_in_history, label='Inlet')
            self.axs[0,1].plot(self.time_history, self.Q_out_history, label='Outlet')
            self.axs[0,1].set_title("Flow Rates")
            self.axs[0,1].set_xlabel("Time [s]")
            self.axs[0,1].set_ylabel("Flow Rate [L/min]")
            self.axs[0,1].legend()

            # Plot velocity profile
            if self.velocity_profiles:
                last_velocity = self.velocity_profiles[-1]
                self.axs[1,0].plot(np.linspace(0, 2500, len(last_velocity)), last_velocity)
                self.axs[1,0].set_title("Velocity Profile")
                self.axs[1,0].set_xlabel("Position [m]")
                self.axs[1,0].set_ylabel("Velocity [m/s]")

            # Plot pressure profile
            if self.pressure_profiles:
                last_pressure = self.pressure_profiles[-1]
                self.axs[1,1].plot(np.linspace(0, 2500, len(last_pressure)), np.array(last_pressure)/1e5)
                self.axs[1,1].set_title("Pressure Profile")
                self.axs[1,1].set_xlabel("Position [m]")
                self.axs[1,1].set_ylabel("Pressure [bar]")

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Plot update error: {e}")

    def run_simulation(self):
        """Main simulation loop (runs in separate thread)"""
        # Initialize simulation parameters from GUI inputs
        try:
            L = float(self.entries["Pipe Length (m)"].get())
            D = float(self.entries["Pipe Diameter (m)"].get())
            bulk_modulus = float(self.entries["Bulk Modulus (GPa)"].get()) * 1e9
            mu_initial = float(self.entries["Initial Viscosity (Pa·s)"].get())
            K = float(self.entries["Consistency Index (K)"].get())
            n = float(self.entries["Flow Index (n)"].get())
        except ValueError as e:
            print(f"Invalid input: {e}")
            return

        # Simulation setup (from original code)
        dx = 100.0
        N = int(L // dx)
        T_max = 100.0
        cfl = 0.8
        rho_0 = 1200.0
        A = np.pi*(D/2)**2

        # Derived parameters
        c = np.sqrt(bulk_modulus/rho_0)
        dt = cfl * dx / (c + 1.0)
        Re_crit = 2100 + 875 * (1 - n)
        Q_schedule = np.array([
            [0, 0/60000], [20, 200/60000], [30, 200/60000],
            [40, 400/60000], [50, 400/60000], [60, 600/60000],
            [70, 600/60000], [80, 800/60000], [90, 800/60000],
            [100, 1000/60000], [110, 1000/60000], [120, 800/60000],
            [130, 800/60000], [140, 600/60000], [150, 600/60000],
            [160, 400/60000], [170, 400/60000], [180, 200/60000],
            [190, 200/60000], [200, 0/60000], [T_max, 0/60000]
        ])

        # Initialize arrays
        x_centers = np.linspace(dx/2, L-dx/2, N)
        x_faces = np.linspace(0, L, N+1)
        u = np.zeros(N+1, dtype=np.float64)
        P = np.full(N, self.P_outlet, dtype=np.float64)
        rho = np.full(N, rho_0, dtype=np.float64)
        mu = mu_initial

        # Matrix initialization
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N-1)
        A_p = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
        alpha = 0.5 * mu_initial * dt / (rho_0 * dx**2)
        main_visc = (1 + 2*alpha) * np.ones(N-1)
        off_visc = -alpha * np.ones(N-2)
        A_visc = diags([off_visc, main_visc, off_visc], [-1, 0, 1], format='csr')

        t = 0.0
        self.time_history = []
        self.P_inlet_history = []
        self.Q_in_history = []
        self.Q_out_history = []
        self.velocity_profiles = []
        self.pressure_profiles = []

        while self.sim_running and t < T_max:
            try:
                # Update boundary conditions with current P_outlet
                with self.lock:
                    current_P_outlet = self.P_outlet
                
                # Velocity boundary conditions
                Q_in = np.interp(t, Q_schedule[:,0], Q_schedule[:,1])
                u[0] = Q_in / A
                u[-1] = u[-2] + (current_P_outlet - P[-1])/(rho_0 * c)
                P[-1] = current_P_outlet

                # Simulation steps from original code
                rho_half = rho - dt * rho_0 * np.diff(u)/dx
                rhs_p = (np.diff(u)/dx - (rho_half - rho_0)/(dt * rho_0))
                A_p_scaled = A_p * (dt/(rho_0 * dx**2))
                P = spsolve(A_p_scaled, rhs_p) + current_P_outlet
                rhs_visc = u[1:-1] + 0.5*mu*dt/rho_0 * (u[2:] - 2*u[1:-1] + u[:-2])/dx**2
                u[1:-1] = spsolve(A_visc, rhs_visc)
                gradP = np.diff(P)/dx
                u[1:-1] -= dt * (gradP/rho_0 + 0.5*mu*u[1:-1]/rho_0/D**2)
                rho = rho_half - dt * rho_0 * np.diff(u)/dx

                # Update viscosity
                Re_MR = (rho_0 * np.average(u)**(2 - n) * D**n) / (K * 8**(n - 1))
                if Re_MR < Re_crit:
                    f = 16 / Re_MR if Re_MR > 0 else 0.0
                else:
                    # Simplified turbulent calculation for real-time
                    f = 0.079 * Re_MR**(-0.25)
                mu = (rho_0 * np.average(u) * D) / Re_MR if Re_MR > 0 else mu_initial
                alpha = 0.5 * mu * dt / (rho_0 * dx**2)
                A_visc = diags([-alpha*np.ones(N-2), (1 + 2*alpha)*np.ones(N-1), -alpha*np.ones(N-2)], [-1, 0, 1], format='csr')

                # Store results
                self.time_history.append(t)
                self.P_inlet_history.append(P[0]/1e5)
                self.Q_in_history.append(Q_in * 60000)
                self.Q_out_history.append(u[-1] * A * 60000)
                self.velocity_profiles.append(u.copy())
                self.pressure_profiles.append(P.copy())

                # Update plots periodically
                if len(self.time_history) % 10 == 0:
                    self.root.after(0, self.update_plots)

                t += dt
                time.sleep(dt/10)  # Slow down simulation for real-time interaction

            except Exception as e:
                print(f"Simulation error: {e}")
                break

        self.sim_running = False
        self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveFluidSimulator(root)
    root.mainloop()