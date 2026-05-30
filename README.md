# Image-Quality-Analyzer-V1

Image Quality Analyzer (v1)

- Uses Laplacian variance for sharpness estimation
- Splits image into patches to generate heatmap
- Visualizes sharpness distribution via overlay
- A few terms to note:
-  Sharp Ratio: What fraction of the image is considered sharp.
-  Standard Deviation(Std): How spread out the values of the image patches are. (Low std means all are generally blurry/uniformly texture, while High std means there are clear focus areas)
  

Run:
python main.py image.jpg[replace with own image file]
