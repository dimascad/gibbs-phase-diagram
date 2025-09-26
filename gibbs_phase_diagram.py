import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar, fsolve
    from mpl_toolkits.mplot3d import Axes3D
    import warnings
    warnings.filterwarnings('ignore')
    return Axes3D, fsolve, minimize_scalar, mo, np, plt, warnings


@app.cell
def __(mo):
    mo.md(r"""
    # Gibbs Free Energy and Phase Diagrams: A 3D Perspective

    This interactive notebook demonstrates how temperature affects Gibbs free energy curves and how the common tangent construction determines phase boundaries.

    ## The 3D Thermodynamic Surface

    Imagine a 3D surface where:
    - **X-axis**: Composition (mole fraction of component B)
    - **Y-axis**: Temperature (K)
    - **Z-axis**: Gibbs free energy (J/mol)

    Each phase (α and β) forms its own 3D surface. The 2D plots you see are horizontal slices through these surfaces at constant temperature.

    ## The Physics Behind It

    The Gibbs free energy for each phase is calculated using the **regular solution model**:

    $$G = G_{reference} + G_{mixing} + G_{excess}$$

    Where:
    - $G_{reference} = x \cdot G°_B + (1-x) \cdot G°_A$ (temperature-dependent)
    - $G_{mixing} = RT[x \ln(x) + (1-x) \ln(1-x)]$ (entropy of mixing)
    - $G_{excess} = x(1-x) \cdot \omega$ (interaction energy)

    Use the slider below to explore different temperature slices of the 3D surface!
    """)
    return


@app.cell
def __(mo):
    temperature_slider = mo.ui.slider(
        start=300,
        stop=1500,
        step=10,
        value=800,
        label="Temperature (K)",
        show_value=True
    )
    temperature_slider
    return (temperature_slider,)


@app.cell
def __(temperature_slider):
    # Model parameters for a simple binary A-B system
    # Using a regular solution model
    
    T = temperature_slider.value
    R = 8.314  # Gas constant J/mol·K
    
    # Interaction parameters (simplified)
    # Different for each phase to create realistic behavior
    omega_alpha = 15000  # J/mol - interaction parameter for alpha phase
    omega_beta = 8000    # J/mol - interaction parameter for beta phase
    
    # Reference Gibbs energies (temperature dependent)
    # These create the baseline difference between phases
    G0_A_alpha = 0
    G0_B_alpha = 5000 - 2 * T  # Temperature dependent
    G0_A_beta = 2000 - 1.5 * T
    G0_B_beta = 3000 - 2.5 * T
    
    # Gibbs free energy function for each phase
    def G_alpha(x):
        """Gibbs free energy of alpha phase"""
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_alpha + (1-x) * G0_A_alpha
        G_excess = x * (1-x) * omega_alpha
        return G_ref + G_mix + G_excess
    
    def G_beta(x):
        """Gibbs free energy of beta phase"""
        if x <= 0 or x >= 1:
            return np.inf
        G_mix = R * T * (x * np.log(x) + (1-x) * np.log(1-x))
        G_ref = x * G0_B_beta + (1-x) * G0_A_beta
        G_excess = x * (1-x) * omega_beta
        return G_ref + G_mix + G_excess
    
    # Derivatives for common tangent calculation
    def dG_alpha_dx(x):
        """Derivative of G_alpha with respect to composition"""
        if x <= 0 or x >= 1:
            return np.inf
        return (G0_B_alpha - G0_A_alpha + R * T * np.log(x/(1-x)) + 
                omega_alpha * (1 - 2*x))
    
    def dG_beta_dx(x):
        """Derivative of G_beta with respect to composition"""
        if x <= 0 or x >= 1:
            return np.inf
        return (G0_B_beta - G0_A_beta + R * T * np.log(x/(1-x)) + 
                omega_beta * (1 - 2*x))
    return (
        G0_A_alpha,
        G0_A_beta,
        G0_B_alpha,
        G0_B_beta,
        G_alpha,
        G_beta,
        R,
        T,
        dG_alpha_dx,
        dG_beta_dx,
        omega_alpha,
        omega_beta,
    )


@app.cell
def __(G_alpha, G_beta, dG_alpha_dx, dG_beta_dx, fsolve, np):
    # Find common tangent points
    def find_common_tangent():
        """Find the common tangent between alpha and beta phases"""
        def tangent_condition(x):
            x1, x2 = x[0], x[1]
            # Ensure valid composition range
            if x1 <= 0.001 or x1 >= 0.999 or x2 <= 0.001 or x2 >= 0.999:
                return [1e10, 1e10]
            
            try:
                # Common tangent conditions:
                # 1. Slopes are equal: dG_alpha/dx at x1 = dG_beta/dx at x2
                # 2. Tangent line connects both points
                slope_diff = dG_alpha_dx(x1) - dG_beta_dx(x2)
                
                # The tangent line equation must be satisfied
                tangent_diff = (G_beta(x2) - G_alpha(x1)) - dG_alpha_dx(x1) * (x2 - x1)
                
                return [slope_diff, tangent_diff]
            except:
                return [1e10, 1e10]
        
        # Try multiple initial guesses to find the solution
        initial_guesses = [
            [0.2, 0.8],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.4, 0.6]
        ]
        
        for guess in initial_guesses:
            try:
                result = fsolve(tangent_condition, guess, full_output=True)
                x_sol = result[0]
                info = result[1]
                
                # Check if solution is valid
                if (0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and
                    x_sol[0] < x_sol[1] and np.sum(info['fvec']**2) < 1e-10):
                    return x_sol[0], x_sol[1], True
            except:
                continue
        
        return None, None, False
    
    x1_tangent, x2_tangent, tangent_found = find_common_tangent()
    return find_common_tangent, tangent_found, x1_tangent, x2_tangent


@app.cell
def __(mo):
    mo.md(r"""
    ## 3D Gibbs Free Energy Surface
    
    Below is the 3D surface showing how Gibbs free energy varies with both composition and temperature. 
    The current temperature slice is highlighted in green.
    """)
    return


@app.cell
def __(
    G0_A_alpha,
    G0_A_beta,
    G0_B_alpha,
    G0_B_beta,
    R,
    T,
    np,
    omega_alpha,
    omega_beta,
    plt,
):
    # Create 3D visualization of the Gibbs surface
    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create meshgrid for composition and temperature
    x_mesh = np.linspace(0.01, 0.99, 50)
    T_mesh = np.linspace(300, 1500, 50)
    X, T_grid = np.meshgrid(x_mesh, T_mesh)
    
    # Calculate Gibbs energy for each phase across all compositions and temperatures
    def calc_G_alpha_3d(x, temp):
        G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G0_B_alpha_temp = 5000 - 2 * temp
        G_ref = x * G0_B_alpha_temp + (1-x) * G0_A_alpha
        G_excess = x * (1-x) * omega_alpha
        return G_ref + G_mix + G_excess
    
    def calc_G_beta_3d(x, temp):
        G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
        G0_A_beta_temp = 2000 - 1.5 * temp
        G0_B_beta_temp = 3000 - 2.5 * temp
        G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
        G_excess = x * (1-x) * omega_beta
        return G_ref + G_mix + G_excess
    
    # Calculate surfaces
    G_alpha_surf = np.zeros_like(X)
    G_beta_surf = np.zeros_like(X)
    
    for i in range(len(T_mesh)):
        for j in range(len(x_mesh)):
            G_alpha_surf[i, j] = calc_G_alpha_3d(X[i, j], T_grid[i, j])
            G_beta_surf[i, j] = calc_G_beta_3d(X[i, j], T_grid[i, j])
    
    # Plot the surfaces
    surf_alpha = ax_3d.plot_surface(X, T_grid, G_alpha_surf, 
                                    cmap='Blues', alpha=0.7, 
                                    linewidth=0, antialiased=True)
    surf_beta = ax_3d.plot_surface(X, T_grid, G_beta_surf, 
                                   cmap='Reds', alpha=0.7, 
                                   linewidth=0, antialiased=True)
    
    # Add current temperature slice
    x_line = np.linspace(0.01, 0.99, 100)
    G_alpha_line = [calc_G_alpha_3d(x, T) for x in x_line]
    G_beta_line = [calc_G_beta_3d(x, T) for x in x_line]
    T_line = np.full_like(x_line, T)
    
    ax_3d.plot(x_line, T_line, G_alpha_line, 'b-', linewidth=3, label='α phase (current T)')
    ax_3d.plot(x_line, T_line, G_beta_line, 'r-', linewidth=3, label='β phase (current T)')
    
    # Highlight the current temperature plane
    xx = np.array([0, 1, 1, 0])
    yy = np.array([T, T, T, T])
    zz = np.array([np.min(G_alpha_surf), np.min(G_alpha_surf), 
                   np.max(G_beta_surf), np.max(G_beta_surf)])
    verts = [list(zip(xx, yy, zz))]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, alpha=0.2, facecolor='green', edgecolor='green')
    ax_3d.add_collection3d(poly)
    
    ax_3d.set_xlabel('Composition (x_B)', fontsize=12)
    ax_3d.set_ylabel('Temperature (K)', fontsize=12)
    ax_3d.set_zlabel('Gibbs Free Energy (J/mol)', fontsize=12)
    ax_3d.set_title('3D Gibbs Free Energy Surface\nCurrent temperature slice shown in green', fontsize=14)
    
    # Set viewing angle
    ax_3d.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    gibbs_3d = plt.gcf()
    plt.close()
    
    gibbs_3d
    return (
        G_alpha_line,
        G_alpha_surf,
        G_beta_line,
        G_beta_surf,
        Poly3DCollection,
        T_grid,
        T_line,
        T_mesh,
        X,
        ax_3d,
        calc_G_alpha_3d,
        calc_G_beta_3d,
        fig_3d,
        gibbs_3d,
        poly,
        surf_alpha,
        surf_beta,
        verts,
        x_line,
        x_mesh,
        xx,
        yy,
        zz,
    )


@app.cell
def __(
    G0_A_alpha,
    G0_A_beta,
    G0_B_alpha,
    G0_B_beta,
    G_alpha,
    G_beta,
    R,
    T,
    dG_alpha_dx,
    dG_beta_dx,
    fsolve,
    mo,
    np,
    omega_alpha,
    omega_beta,
    plt,
    tangent_found,
    x1_tangent,
    x2_tangent,
):
    # Create figure with side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT PLOT: Gibbs free energy curves
    # Composition range
    x_range = np.linspace(0.001, 0.999, 500)
    
    # Calculate Gibbs energies
    G_alpha_vals = [G_alpha(x) for x in x_range]
    G_beta_vals = [G_beta(x) for x in x_range]
    
    # Plot the curves
    ax1.plot(x_range, G_alpha_vals, 'b-', linewidth=2, label='α phase')
    ax1.plot(x_range, G_beta_vals, 'r-', linewidth=2, label='β phase')
    
    # Plot common tangent if found
    if tangent_found:
        # Calculate tangent line
        y1 = G_alpha(x1_tangent)
        slope = dG_alpha_dx(x1_tangent)
        x_tangent = np.array([0, 1])
        y_tangent = y1 + slope * (x_tangent - x1_tangent)
        
        ax1.plot(x_tangent, y_tangent, 'g--', linewidth=2, label='Common tangent')
        ax1.plot(x1_tangent, y1, 'go', markersize=8)
        ax1.plot(x2_tangent, G_beta(x2_tangent), 'go', markersize=8)
        
        # Add vertical lines to show phase compositions
        ax1.axvline(x=x1_tangent, color='g', linestyle=':', alpha=0.5)
        ax1.axvline(x=x2_tangent, color='g', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Composition (x_B)', fontsize=12)
    ax1.set_ylabel('Gibbs Free Energy (J/mol)', fontsize=12)
    ax1.set_title(f'Gibbs Free Energy at T = {T} K', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_vals = G_alpha_vals + G_beta_vals
    finite_vals = [v for v in all_vals if np.isfinite(v)]
    if finite_vals:
        y_min, y_max = min(finite_vals), max(finite_vals)
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # RIGHT PLOT: Phase diagram
    # Calculate phase diagram by finding common tangents at different temperatures
    temperatures = np.linspace(300, 1500, 50)
    x1_values = []
    x2_values = []
    valid_temps = []
    
    for temp in temperatures:
        # Recalculate temperature-dependent parameters
        G0_B_alpha_temp = 5000 - 2 * temp
        G0_A_beta_temp = 2000 - 1.5 * temp
        G0_B_beta_temp = 3000 - 2.5 * temp
        
        def G_alpha_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_alpha_temp + (1-x) * G0_A_alpha
            G_excess = x * (1-x) * omega_alpha
            return G_ref + G_mix + G_excess
        
        def G_beta_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            G_mix = R * temp * (x * np.log(x) + (1-x) * np.log(1-x))
            G_ref = x * G0_B_beta_temp + (1-x) * G0_A_beta_temp
            G_excess = x * (1-x) * omega_beta
            return G_ref + G_mix + G_excess
        
        def dG_alpha_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_alpha_temp - G0_A_alpha + R * temp * np.log(x/(1-x)) + 
                    omega_alpha * (1 - 2*x))
        
        def dG_beta_dx_temp(x):
            if x <= 0 or x >= 1:
                return np.inf
            return (G0_B_beta_temp - G0_A_beta_temp + R * temp * np.log(x/(1-x)) + 
                    omega_beta * (1 - 2*x))
        
        def tangent_condition(x):
            x1, x2 = x[0], x[1]
            if x1 <= 0.001 or x1 >= 0.999 or x2 <= 0.001 or x2 >= 0.999:
                return [1e10, 1e10]
            
            try:
                slope_diff = dG_alpha_dx_temp(x1) - dG_beta_dx_temp(x2)
                tangent_diff = (G_beta_temp(x2) - G_alpha_temp(x1)) - dG_alpha_dx_temp(x1) * (x2 - x1)
                return [slope_diff, tangent_diff]
            except:
                return [1e10, 1e10]
        
        # Try to find common tangent
        for guess in [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]]:
            try:
                result = fsolve(tangent_condition, guess, full_output=True)
                x_sol = result[0]
                info = result[1]
                
                if (0.001 < x_sol[0] < 0.999 and 0.001 < x_sol[1] < 0.999 and
                    x_sol[0] < x_sol[1] and np.sum(info['fvec']**2) < 1e-10):
                    x1_values.append(x_sol[0])
                    x2_values.append(x_sol[1])
                    valid_temps.append(temp)
                    break
            except:
                continue
    
    if valid_temps:
        # Plot the binodal curve
        ax2.plot(x1_values, valid_temps, 'b-', linewidth=2, label='α phase boundary')
        ax2.plot(x2_values, valid_temps, 'r-', linewidth=2, label='β phase boundary')
        
        # Fill the two-phase region
        ax2.fill_betweenx(valid_temps, x1_values, x2_values, alpha=0.3, color='gray', label='α + β')
        
        # Add labels for single-phase regions
        if x1_values:
            ax2.text(0.1, np.mean(valid_temps), 'α', fontsize=16, ha='center')
            ax2.text(0.9, np.mean(valid_temps), 'β', fontsize=16, ha='center')
        
        # Mark current temperature
        ax2.axhline(y=T, color='green', linestyle='--', linewidth=2, label=f'Current T = {T} K')
        
        # Mark current phase compositions if they exist
        if tangent_found:
            ax2.plot(x1_tangent, T, 'go', markersize=10)
            ax2.plot(x2_tangent, T, 'go', markersize=10)
    
    ax2.set_xlabel('Composition (x_B)', fontsize=12)
    ax2.set_ylabel('Temperature (K)', fontsize=12)
    ax2.set_title('Phase Diagram', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(300, 1500)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot = plt.gcf()
    plt.close()
    
    combined_plot
    return (
        G0_A_beta_temp,
        G0_B_alpha_temp,
        G0_B_beta_temp,
        G_alpha_temp,
        G_alpha_vals,
        G_beta_temp,
        G_beta_vals,
        all_vals,
        ax1,
        ax2,
        combined_plot,
        dG_alpha_dx_temp,
        dG_beta_dx_temp,
        fig,
        finite_vals,
        guess,
        info,
        result,
        slope,
        slope_diff,
        tangent_condition,
        tangent_diff,
        temp,
        temperatures,
        valid_temps,
        x1,
        x1_values,
        x2,
        x2_values,
        x_range,
        x_sol,
        x_tangent,
        y1,
        y_max,
        y_min,
        y_range,
        y_tangent,
    )


@app.cell
def __(mo, tangent_found, x1_tangent, x2_tangent):
    mo.md(f"""
    ## Current State Summary

    At the selected temperature:
    - Common tangent found: {"Yes" if tangent_found else "No"}
    {f"- α phase composition: x_B = {x1_tangent:.3f}" if tangent_found else ""}
    {f"- β phase composition: x_B = {x2_tangent:.3f}" if tangent_found else ""}

    The common tangent construction determines the equilibrium compositions of coexisting phases. When two phases coexist at equilibrium, they must have equal chemical potentials, which is represented geometrically by the common tangent line touching both Gibbs curves.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## Understanding the 3D → 2D Connection
    
    ### What You're Seeing:
    
    1. **The 3D Surface (above)**: Shows the complete Gibbs free energy landscape. Each phase creates a curved surface in 3D space. The green plane shows where we're "slicing" at the current temperature.
    
    2. **The 2D Plots (middle)**: These are the intersection of the green plane with the 3D surfaces - like cutting through a mountain and looking at the cross-section.
    
    3. **The Phase Diagram (right)**: Maps out all the common tangent points across all temperatures, creating the phase boundaries.
    
    ### The Mathematical Magic:
    
    At each temperature, we solve for where:
    - The slopes (chemical potentials) are equal: $\frac{\partial G_\alpha}{\partial x} = \frac{\partial G_\beta}{\partial x}$
    - A single tangent line connects both curves
    
    This gives us the equilibrium compositions where both phases can coexist!
    
    ### Try This:
    
    - **Low temperatures** (300-600K): Strong phase separation due to high interaction energy
    - **Medium temperatures** (700-1000K): Moderate miscibility gap
    - **High temperatures** (>1100K): Increased mixing due to entropy dominating
    
    Notice how the 3D surfaces get "flatter" at high temperatures - this is entropy smoothing out the energy differences!
    """)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run() 
# Added attribution
