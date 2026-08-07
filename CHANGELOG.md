<!-- # Image-Quality-Analyzer

An image quality assessment tool that combines classical computer vision techniques with machine learning to evaluate image sharpness and classify images as Sharp or Blurry, detect faces and eye states, and provides feedback on the image, such as saturation and colour calance and lighting.



# Project Vision

### Build a hybrid ML-powered image quality analyzer that evaluates photographs in a way that is understandable and useful to photographers, providing objective measurements, intelligent explanations, and actionable recommendations for improving image quality.

## Core Objectives
- Evaluate image sharpness accurately.  
- Detect different causes of blur (motion, defocus, etc.).
- Assess image noise.
- Evaluate exposure and tonal balance.
- Assess colour quality (white balance, saturation, contrast).
- Analyse important subjects (faces, eyes, text).
- Produce human-readable reports through the site.
- Provide practical feedback and recommendations rather than raw numbers.

## Guiding Principle

Every new feature must help answer one question: "Does this help a photographer better understand or improve their image?"



## Image Quality Analyzer 

Now, we added a simple Machine Learning feature, known as the Random Forest Classifier. Along it, we have also introduced an additional sharpness parameter known as wavelets.

So what does each parameter do?

- `Laplacian Variance`: First parameter. Locally, it measures the difference in pixel intensity between neighboring pixels, and then outputs a single score globally through variance. 

- `FFT High Frequency Ratio`: Uses the Fast Fourier Transform (FFT) to measure the proportion of high-frequency information present across the entire image. Sharp images generally contain more high-frequency content than blurry images. 

- `Wavelet Energy Ratio`: Captures localized high frequency detail while preserving spatial information, allowing the algorithm to measure both the amount of fine detail and where it is located within the image.

- `Consistency`: Measures how evenly sharpness is distributed along image.

- `Exposure` : Penalizes under or   overexposed images.



### Machine Learning Pipeline

Version 3 introduces a supervised learning workflow:
1. We first manually label images as sharp or blurry.
2. The handcrafted features are extracted from each image.
3. A dataset is generated automatically.
4. A Random Forest Classifier is trained on the extracted features.
5. The trained model is saved as `model.pkl`.
6. New images are then classified automatically with prediction probabilities.

Example Output

- Prediction          : Sharp 
- Probability (Sharp) : 95% 
- Probability (Blurry): 5%

___________________________________________________________

### Running the Project

- Analyze and compare images:   ```python main.py```

- Generate a labelled dataset:  ```python ml/generate_dataset.py```

- Train the Random Forest model: ```python -m ml.train_model```

- Predict image quality:  ``` python -m ml.predict ```


```text
                Input Image
                     │
                     ▼
          Feature Extraction
                     │
     ┌───────────────┼───────────────┐
     │               │               │
Laplacian        FFT Ratio      Wavelet Ratio
     │               │               │
     ├───────────────┼───────────────┤
     │               │               │
Sharp Ratio    Consistency      Exposure
                     │
                     ▼
      Random Forest Classifier
                     │
                     ▼
      Prediction + Confidence
``` -->