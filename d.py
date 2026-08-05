import cv2
import numpy as np
import matplotlib.pyplot as plt
from detection.face_analysis_SCRFD import detect_faces
img_path = r"image_cache\Excellent\001_98_fra-728.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = detect_faces(img)
landmarks = result["face_landmarks"][0]

# Draw all 106 landmarks with their index numbers
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.imshow(img_rgb)

for i, (x, y) in enumerate(landmarks):
    ax.plot(x, y, 'r.', markersize=4)
    ax.annotate(str(i), (x, y), fontsize=5, color='yellow',
                ha='center', va='bottom')

ax.set_title("All 106 Landmarks with Indices")
plt.tight_layout()
plt.savefig("landmarks_visualised.png", dpi=150)
plt.show()
print("Saved to landmarks_visualised.png")