# Matplotlib

Matplotlib is a low level graph plotting library in python that serves as a visualization utility.
Matplotlib was created by John D. Hunter.
Matplotlib is open source and we can use it freely.
Matplotlib is mostly written in python, a few segments are written in C, Objective-C and Javascript for Platform compatibility.

repository : https://github.com/matplotlib/matplotlib 

Matplotlib occupies a peculiar position in the scientific Python ecosystem. It is simultaneously the oldest general-purpose plotting library in wide use and the substrate on which most newer libraries (seaborn, pandas plotting, xarray, scikit-learn's `plot_*` helpers) are built. Understanding it well therefore pays a compound return: you gain direct control over publication figures, and you gain the ability to debug and extend the higher-level tools that delegate to it. This document develops the library from its object model outward, because nearly every difficulty users encounter, from inexplicable layouts to labels that vanish under `tight_layout`, traces back to a misunderstanding of how figures, axes, and artists relate to one another.

## The Object Model

A matplotlib figure is a tree. At the root sits the `Figure`, a canvas with a coordinate system measured in inches but rendered at some DPI. Beneath the figure sit one or more `Axes`, each of which is a rectangular region with its own coordinate systems, its own data limits, and its own collection of child artists. The `Axes` is the object that does the plotting; the `Figure` merely holds axes and orchestrates rendering. This distinction matters because beginners routinely conflate the two, calling `plt.plot` and expecting a global canvas, when in fact pyplot is maintaining a current figure and a current axes on their behalf.

Everything visible on a figure is an `Artist`. Lines, text, patches, images, tick marks, spines, and legends all derive from the `Artist` base class. An artist has properties (color, linewidth, alpha, zorder, visibility), a transform that maps its coordinates into display space, and a `draw` method that the renderer invokes. When you call `fig.savefig` or the GUI backend repaints, the figure walks its artist tree and asks each artist to render itself. This uniformity is what makes matplotlib extensible: a custom artist is simply a class that knows how to draw itself given a renderer and a bounding box.

The practical consequence of the tree is that every plotting call mutates some node. `ax.plot(x, y)` creates a `Line2D` artist and appends it to `ax.lines`. `ax.scatter` creates a `PathCollection` and appends it to `ax.collections`. `ax.set_title` creates or updates a `Text` artist stored in `ax.title`. Once you internalize this, the library stops being a set of magic functions and becomes a mutable scene graph you can inspect and manipulate. `ax.get_children()` enumerates the artists; `line.set_color('red')` modifies one after the fact; `line.remove()` detaches it.

## The pyplot Interface and Its Relationship to the Object API

`matplotlib.pyplot` is a state machine layered on top of the object API. It maintains a current figure (`plt.gcf()`) and a current axes (`plt.gca()`), creating them on demand. `plt.plot(...)` is shorthand for `plt.gca().plot(...)`. This convenience is genuinely useful for exploratory work and short scripts, where the ceremony of `fig, ax = plt.subplots()` feels excessive. It becomes a liability in anything longer than a few dozen lines, because implicit state makes behavior depend on execution history. A function that calls `plt.title` will title whatever axes happens to be current, which may or may not be the axes the caller intended.

The disciplined approach is to use the object API explicitly and treat pyplot as a figure factory. `fig, ax = plt.subplots()` returns both objects, and every subsequent call operates on `ax` or `fig` directly. This pattern composes: functions accept an `ax` argument and draw into it, which makes them testable, reusable in grids, and immune to state leakage. The pyplot-style calls remain available for interactive sessions, but production code should pass axes explicitly. The two styles interoperate because pyplot's current-axes pointer is just a reference into the same object tree.