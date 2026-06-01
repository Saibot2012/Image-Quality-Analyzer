# Image-Quality-Analyzer

Image Quality Analyzer (v1)

 Uses Laplacian variance for sharpness estimation, splits image into patches to generate heatmap, and visualizes sharpness distribution via overlay.
 
 A few terms to note:
  
 - Sharp Ratio: What fraction of the image is considered sharp.
 
 - Standard Deviation(Std): How spread out the values of the image patches are. (Low std means all are generally blurry/uniformly texture, while High std means there are clear focus areas.)

Image Quality Analyzer (v2)

In this upgraded version, we incorporated multi-image comparison, a consistency metric, and exposure awareness. By adding them together, we form a combined score for images which we can use to
rank them:

```python
score = (
    0.4 * norm_sharpness +
    0.25 * sharp_ratio +
    0.15 * consistency +
    0.2 * exposure
        )
```
Where:

- `norm_sharpness` measures global image sharpness using Laplacian variance.
- `sharp_ratio` measures the fraction of image patches considered sharp.
- `consistency` measures how evenly sharpness is distributed.
- `exposure` penalizes over- and under-exposed images

![Results](results/results.png)

 Run:
python main.py image.jpg[replace with own image file]
