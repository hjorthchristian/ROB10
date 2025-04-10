import numpy as np

# Load the NPZ file
data = np.load('color_to_base_tf/transformed_points_homogeneous.npy')
# Open a text file for writing
with open('output.txt', 'w') as f:
    # Check if this is a .npz file with multiple arrays
    if hasattr(data, 'files'):
        # Loop through all the arrays in the NPZ file
        for key in data.files:
            # Write the name of the array
            f.write(f"Array: {key}\n")
            
            # Get the array
            array = data[key]
            
            # Write array shape information
            f.write(f"Shape: {array.shape}\n")
            
            # Write the array data
            # For 1D arrays
            if len(array.shape) == 1:
                f.write(np.array2string(array, separator=', ') + '\n\n')
            # For 2D arrays
            elif len(array.shape) == 2:
                for row in array:
                    f.write(np.array2string(row, separator=', ') + '\n')
                f.write('\n')
            # For higher-dimensional arrays
            else:
                f.write(str(array) + '\n\n')
    else:
        # This is a single array saved with np.save()
        f.write(f"Shape: {data.shape}\n")
        
        # Write the array data
        if len(data.shape) == 1:
            f.write(np.array2string(data, separator=', ') + '\n')
        elif len(data.shape) == 2:
            for row in data:
                f.write(np.array2string(row, separator=', ') + '\n')
        else:
            f.write(str(data) + '\n')

print("Conversion complete!")