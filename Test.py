import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the saved model
state_dict = torch.load('Image-classification-VGG16/trained.pt')

# Create an instance of the model with the same architecture
model = models.vgg16(pretrained=False)
num_classes = 4
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Load the saved parameters into the model
model.load_state_dict(state_dict)

# Put the model in evaluation mode
model.eval()

# Set up the data transforms for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the input image
img = Image.open('Image-classification-VGG16/rt.jpeg')

# Convert the image to RGB if it's grayscale
if img.mode != 'RGB':
    img = img.convert('RGB')

# Apply the transforms to the input image
img_tensor = transform(img)

# Add an extra dimension to the tensor to represent batch size of 1
img_tensor = img_tensor.unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1)
    
# Print the predicted class
print(predicted.item())

# Define the class names
class_names = ['Hoodie', 'Pant','Tshirt', 'Short']

# Print the predicted class name
predicted_class = class_names[predicted.item()]
print(predicted_class)

# Define the true class
true_class = 'Tshirt'

# Compute the accuracy
accuracy = accuracy_score([true_class], [predicted_class])
print(f"Accuracy: {accuracy}")

# Compute the confusion matrix
cm = confusion_matrix([true_class], [predicted_class], labels=class_names)
print("Confusion Matrix:")
print(cm)

# Display the confusion matrix as a heatmap using seaborn
sns.set()
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()
