import torch
import torch.optim as optim
from torch.nn import functional as F
from model import ResNet50
from data import get_data_loaders

def train(model, device, trainloader, optimizer, num_epochs=10):
    model.train()
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0

        for i, (image, label) in enumerate(trainloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(image)
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if (i+1) % 2 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy is :{:.4f}%' 
                    .format(epoch+1, num_epochs, i+1, len(trainloader), loss.item(), 100*correct/total))

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total

        losses.append(epoch_loss)
    return losses

def test(model, device, testloader):
    model.eval()

    correct = 0.0 # 模型正确率
    test_loss = 0.0 # 模型损失值
    total = 0

    with torch.no_grad():
        for image, label in testloader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            test_loss += F.cross_entropy(output, label).item()

            predict = output.argmax(dim=1)
            total += label.size(0)
            correct += (predict == label).sum().item()
            
        avg_loss = test_loss / len(testloader)
        accuracy = 100 * correct / total
        print(f"Test Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50().to(device)
    print('------DEVICE------: ', device)
    
    optimizer = optim.Adam(model.parameters())

    root = './data'

    trainloader, testloader = get_data_loaders(root, root, batch_size=100)

    train(model, device, trainloader, optimizer, num_epochs=10)
    print('------train------: complete')
    test(model, device, testloader)
    print('------test------: complete')
    print('Good job!')
