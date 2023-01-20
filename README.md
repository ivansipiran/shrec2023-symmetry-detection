# SHREC 2023: Symmetry Detection on Point Clouds

# Dataset
The data can be downloaded in this [link](https://drive.google.com/drive/folders/1d27oYoJuWiOZqzwQx6WB9qfybghCr3pX?usp=sharing). The training set contains the point clouds and the ground-truth files. Each ground-truth file contains the number of symmetries for a given point cloud. For the symmetries, each line contains three float numbers for the normal and three float numbers for the point-in-plane. The test dataset contains only the point clouds.

# Tool
In this repository, you can find a tool to visualize the point clouds and their symmetries. This tool is intended to be used for inspection of the dataset and to check results. The only packages required to run this tool are [Numpy](https://numpy.org/) and [Polyscope](https://polyscope.run/py/). If you want to visualize a point cloud and their symmetries, you can run this command:

~~~
> python  view_symmetry.py --path=benchmark-train/ --id=1589
~~~

For example, the above command will show the point cloud  "points1589.txt" from the training dataset. The ground-truth is automatically loaded from the provided ID.