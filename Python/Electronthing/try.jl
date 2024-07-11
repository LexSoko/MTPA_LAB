using GLMakie
using AbstractPlotting

# Generate or load your 3D volume data (replace with actual data)
volume_data = rand(100, 100, 100)

# Set up a scene
scene = Scene(resolution = (800, 600))

# Create a volume plot
volume_plot = Volume(scene, volume_data)

# Customize appearance:
# - Change colormap (e.g., "viridis")
# - Adjust opacity (0.5 for semi-transparent)
# - Set lighting (true for realistic shading)
# - Choose rendering style (isosurface or slices)
set_theme!(scene, Theme(backgroundcolor = :white))
colormap!(volume_plot, :viridis)
opacity!(volume_plot, 0.5)
lighting!(volume_plot, true)
renderstyle!(volume_plot, :isosurface)  # or :slices

# Display the scene
display(scene)
