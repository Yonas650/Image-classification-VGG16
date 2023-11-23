import torch
import torch.nn as nn
import torchvision.models as models #contains VGG16 
import torchvision.transforms as transforms #Image transformation
import torchvision.datasets as datasets # provides pre-loaded and pre-processed datasets such as ImageNet
import ssl #Secure Sockets Layer

ssl._create_default_https_context = ssl._create_unverified_context #disables certificate verification for HTTPS 

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0 #cumulative loss
    correct = 0 #correct predictions
    total = 0 #total number of predictions
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad() #avoids accumulation of gradients from previous batches
        outputs = model(inputs) #The model's forward pass
        loss = criterion(outputs, labels)#loss calculation
        loss.backward()#calcualtion of gradient of the loss
        optimizer.step() # updates the model parameters
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), 100 * correct / total #returns average loss and accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), 100 * correct / total

def main():
    # Set device to GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained VGG model
    vgg = models.vgg16(pretrained=True)
    vgg.to(device)

    # Modify the last layer of the model to output the desired number of classes
    num_classes = 4
    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
    )
    vgg.to(device)

    # Set up data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load data
    train_data = datasets.ImageFolder('train/', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = datasets.ImageFolder('val/', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)#stochastic gradient descent 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)#reduces the learning rate when the validation loss does not improve

    # Train the model
    num_epochs = 10
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(vgg, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(vgg, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vgg.state_dict(), 'trained.pt')
            print('Model saved successfully.')

if __name__ == "__main__":
    main()
