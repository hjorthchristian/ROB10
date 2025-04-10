from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

# -------------------------------
# Parameters
# -------------------------------

# Pallet dimensions
pallet_x, pallet_y = 4, 2
max_height = 10

# Boxes: (length, width, height)
boxes = [(2, 2, 1), (2, 1, 2), (3, 2, 1), (2, 2, 1), (2, 1, 3), (3, 2, 1)]
n = len(boxes)

# -------------------------------
# Optimization with OR-Tools
# -------------------------------

model = cp_model.CpModel()
x, y, z = [], [], []
r = []  # rotation variable
length_vars, width_vars = [], []

for i in range(n):
    l, w, h_i = boxes[i]
    x_i = model.NewIntVar(0, pallet_x, f'x_{i}')
    y_i = model.NewIntVar(0, pallet_y, f'y_{i}')
    z_i = model.NewIntVar(0, max_height, f'z_{i}')
    r_i = model.NewBoolVar(f'r_{i}')  # 0 = no rotation, 1 = 90 deg rotation

    l_i = model.NewIntVar(0, pallet_x, f'l_{i}')
    w_i = model.NewIntVar(0, pallet_y, f'w_{i}')
    model.Add(l_i == l).OnlyEnforceIf(r_i.Not())
    model.Add(l_i == w).OnlyEnforceIf(r_i)
    model.Add(w_i == w).OnlyEnforceIf(r_i.Not())
    model.Add(w_i == l).OnlyEnforceIf(r_i)

    model.Add(x_i + l_i <= pallet_x)
    model.Add(y_i + w_i <= pallet_y)

    x.append(x_i)
    y.append(y_i)
    z.append(z_i)
    r.append(r_i)
    length_vars.append(l_i)
    width_vars.append(w_i)

# No-overlap constraints using Boolean variables
for i in range(n):
    for j in range(i + 1, n):
        b1 = model.NewBoolVar(f'no_overlap_x1_{i}_{j}')
        b2 = model.NewBoolVar(f'no_overlap_x2_{i}_{j}')
        b3 = model.NewBoolVar(f'no_overlap_y1_{i}_{j}')
        b4 = model.NewBoolVar(f'no_overlap_y2_{i}_{j}')
        b5 = model.NewBoolVar(f'no_overlap_z1_{i}_{j}')
        b6 = model.NewBoolVar(f'no_overlap_z2_{i}_{j}')

        model.Add(x[i] + length_vars[i] <= x[j]).OnlyEnforceIf(b1)
        model.Add(x[i] + length_vars[i] > x[j]).OnlyEnforceIf(b1.Not())

        model.Add(x[j] + length_vars[j] <= x[i]).OnlyEnforceIf(b2)
        model.Add(x[j] + length_vars[j] > x[i]).OnlyEnforceIf(b2.Not())

        model.Add(y[i] + width_vars[i] <= y[j]).OnlyEnforceIf(b3)
        model.Add(y[i] + width_vars[i] > y[j]).OnlyEnforceIf(b3.Not())

        model.Add(y[j] + width_vars[j] <= y[i]).OnlyEnforceIf(b4)
        model.Add(y[j] + width_vars[j] > y[i]).OnlyEnforceIf(b4.Not())

        model.Add(z[i] + boxes[i][2] <= z[j]).OnlyEnforceIf(b5)
        model.Add(z[i] + boxes[i][2] > z[j]).OnlyEnforceIf(b5.Not())

        model.Add(z[j] + boxes[j][2] <= z[i]).OnlyEnforceIf(b6)
        model.Add(z[j] + boxes[j][2] > z[i]).OnlyEnforceIf(b6.Not())

        model.AddBoolOr([b1, b2, b3, b4, b5, b6])


# Stability constraints
for i in range(n):
    support_conditions = []

    # Add boolean var for support on pallet
    support_on_pallet = model.NewBoolVar(f'support_pallet_{i}')
    model.Add(z[i] == 0).OnlyEnforceIf(support_on_pallet)
    model.Add(z[i] != 0).OnlyEnforceIf(support_on_pallet.Not())
    support_conditions.append(support_on_pallet)

    for j in range(n):
        if i == j:
            continue

        support_x = model.NewBoolVar(f'support_x_{i}_{j}')
        support_y = model.NewBoolVar(f'support_y_{i}_{j}')
        support_z = model.NewBoolVar(f'support_z_{i}_{j}')
        support_all = model.NewBoolVar(f'supported_{i}_{j}')

        model.Add(x[i] >= x[j]).OnlyEnforceIf(support_x)
        model.Add(x[i] + length_vars[i] <= x[j] + length_vars[j]).OnlyEnforceIf(support_x)
        model.Add(y[i] >= y[j]).OnlyEnforceIf(support_y)
        model.Add(y[i] + width_vars[i] <= y[j] + width_vars[j]).OnlyEnforceIf(support_y)
        model.Add(z[i] == z[j] + boxes[j][2]).OnlyEnforceIf(support_z)

        model.AddBoolAnd([support_x, support_y, support_z]).OnlyEnforceIf(support_all)
        model.AddBoolOr([support_x.Not(), support_y.Not(), support_z.Not()]).OnlyEnforceIf(support_all.Not())

        support_conditions.append(support_all)

    model.AddBoolOr(support_conditions)

# Objective: minimize total height
max_z = model.NewIntVar(0, max_height, 'max_z')
for i in range(n):
    model.Add(max_z >= z[i] + boxes[i][2])
model.Minimize(max_z)

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# -------------------------------
# Visualization with Matplotlib
# -------------------------------

def plot_box(ax, origin, size, color):
    x0, y0, z0 = origin
    dx, dy, dz = size

    # Vertices
    verts = [
        [x0, y0, z0],
        [x0 + dx, y0, z0],
        [x0 + dx, y0 + dy, z0],
        [x0, y0 + dy, z0],
        [x0, y0, z0 + dz],
        [x0 + dx, y0, z0 + dz],
        [x0 + dx, y0 + dy, z0 + dz],
        [x0, y0 + dy, z0 + dz],
    ]

    # Faces
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # bottom
        [verts[4], verts[5], verts[6], verts[7]],  # top
        [verts[0], verts[1], verts[5], verts[4]],  # front
        [verts[1], verts[2], verts[6], verts[5]],  # right
        [verts[2], verts[3], verts[7], verts[6]],  # back
        [verts[3], verts[0], verts[4], verts[7]],  # left
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=0.8))

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Minimized height: {solver.Value(max_z)}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, max(pallet_x, pallet_y))
    ax.set_ylim(0, max(pallet_y, pallet_x))
    ax.set_zlim(0, max_height)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("TETRIS Depalletized Box Stacking")

    cmap = cm.get_cmap('tab20', n)
    for i in range(n):
        box_pos = (solver.Value(x[i]), solver.Value(y[i]), solver.Value(z[i]))
        box_size = (solver.Value(length_vars[i]), solver.Value(width_vars[i]), boxes[i][2])
        plot_box(ax, box_pos, box_size, cmap(i))

    # Optional: draw pallet base
    ax.bar3d(0, 0, 0, pallet_x, pallet_y, 0.1, color='gray', alpha=0.3, shade=True)

    plt.show()
else:
    print("No feasible solution found.")
