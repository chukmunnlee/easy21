n = 2000

def func_vec(x1s, x2s):
    return x1s * x1s + 4 * x2s * x2s

np.random.seed()
x1s = np.random.uniform(-1, 1, n)
x2s = np.random.uniform(-1, 1, n)
ys = func_vec(x1s, x2s)

fig = plt.figure(22)

# Scatter
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x1s, x2s, color = 'g', s = 2, edgecolor = 'none')
ax1.set_ylim([-1,1])
ax1.set_xlim([-1,1])

# Contour
xi = np.linspace(-1,1,20)
yi = np.linspace(-1,1,20)
zi = griddata((x2s, x1s), ys, (xi[None,:], yi[:,None]), method = 'cubic')
ax1.contour(xi, yi, zi, 6, linewidths = 1, colors = ('#0000ff', '#0099ff', '#009999', '#999900', '#ff9900', '#ff0000'))

# 3D visualization
ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
X, Y = np.meshgrid(xi, yi)
ax2.plot_wireframe(X, Y, zi, rstride = 1, cstride = 1)
ax2.view_init(28, -144) 

plt.show()
