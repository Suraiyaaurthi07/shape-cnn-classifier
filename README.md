# shape-cnn-classifier
## Results

**Standard test set accuracy: 100.00% (300 / 300)**

| Class | Accuracy | Precision | Recall | F1‑score |
|---|---|---|---|---|
| Circle | 100.00% (100/100) | 1.0000 | 1.0000 | 1.0000 |
| Square | 100.00% (100/100) | 1.0000 | 1.0000 | 1.0000 |
| Triangle | 100.00% (100/100) | 1.0000 | 1.0000 | 1.0000 |

**Custom hand-drawn photo accuracy: 4 / 10 = 40.0%**

The model reaches perfect accuracy on the synthetic test set, which is drawn from the same procedurally-generated distribution as training data (jittered outlines with Gaussian noise on a plain background). On real hand-drawn photos, however, accuracy drops sharply to 40%, exposing a significant domain gap. Triangles were recognized reliably (4/4), but every circle was misclassified as a square, and every square was misclassified as a triangle. This suggests the model latched onto low-level cues (like corner/edge patterns and stroke texture) that hold for clean synthetic outlines but don't transfer to photos of real drawings, which have uneven lighting, background clutter, and imperfect hand-drawn geometry.

---

## Plots

**Training History** - Loss vs. Epochs and Accuracy vs. Epochs for training and validation sets.

<img width="1639" height="539" alt="image" src="https://github.com/user-attachments/assets/91359e04-59d3-4abd-bdca-0f916ef65a86" />


**Confusion Matrix** - heatmap of predictions vs. true labels on the standard test set.

<img width="720" height="607" alt="image" src="https://github.com/user-attachments/assets/54b7c0a3-2929-4382-9a8f-a63a68660a44" />


**Custom Prediction Gallery** - all 10 hand-drawn photos with predicted class and confidence (e.g. "Pred: Triangle (99.1%)"), green for correct and red for incorrect.

<img width="1396" height="629" alt="image" src="https://github.com/user-attachments/assets/68a4f745-d34b-48a3-9c16-4a1e2cd7c5ed" />


**Visual Error Analysis** - not applicable for the standard test set: the model classified all 300 test images correctly (0 misclassified), so no error samples exist to plot.

---

## Architecture

A custom CNN (`class CNN(nn.Module)`) with three convolutional blocks followed by a fully connected classifier head:

| Layer | Details |
|---|---|
| Conv Block 1 | `Conv2d(1 → 32, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2,2)` |
| Conv Block 2 | `Conv2d(32 → 64, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2,2)` |
| Conv Block 3 | `Conv2d(64 → 128, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2,2)` |
| Flatten | `128 × 16 × 16 → 32768` |
| FC1 | `Linear(32768 → 256)` → `ReLU` |
| Dropout | `p = 0.4` |
| FC2 (output) | `Linear(256 → 3)` |

Input images are 128×128 grayscale (single-channel). Three conv+pool stages progressively shrink the spatial resolution from 128×128 down to 16×16 while increasing channel depth from 1 to 128. Dropout (p=0.4) is applied before the final linear layer to reduce overfitting. Trained with `nn.CrossEntropyLoss` and the `Adam` optimizer (lr=0.001) for 10 epochs, batch size 64, on a synthetic training set of 500 procedurally-generated images per class (Circle, Square, Triangle) with random rotation augmentation.


