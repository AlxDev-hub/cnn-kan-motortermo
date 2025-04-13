from efficient_kan import KAN
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado: ", device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose(
    [transforms.Resize(size = (164,164)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

dataset = datasets.ImageFolder('output_folder', transform = transform)
print("Informações sobre o Dataset: \n\n", dataset)
print("\nRótulos: ", dataset.class_to_idx)

def ekan_model():
    model = KAN([164*164*3, 164, 64, 32, 2])
    return model

num_epoch = 100
learning_rate = 0.001

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

results_acc = {}
results_precision = {}
results_recall = {}
results_f1 = {}

training_start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f'\nFold {fold+1}/{k_folds}')
    print(f'\nQuantidade de dados (Treinamento): {len(train_idx)}')
    print(f'Quantidade de dados (Teste): {len(test_idx)}')

    model = ekan_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size=len(train_idx), sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size=len(test_idx), sampler=test_sampler)

    all_targets = []
    for img, rtl, in trainloader:
        all_targets.extend(rtl.tolist())

    inlier, outlier = [], []
    for i in range(len(train_sampler)):
        if all_targets[i] == 0:
            inlier.append(all_targets[i])
        elif all_targets[i] == 1:
            outlier.append(all_targets[i])

    print("\n!!!Distribuição dos dados de treinamento!!!\n")
    print(f'Inliers: {len(inlier)}')
    print(f'Outliers: {len(outlier)}\n')

    train_losses = []
    train_acc = []

    for epoch in range(num_epoch):
        model.train()
        running_train_loss=0.0
        total_samples = 0
        all_preds_train = []
        all_labels_train = []

        for inputs_train, labels_train in trainloader:
            inputs_train = inputs_train.view(-1, 164*164*3).to(device)
            labels_train = labels_train.to(device)
    
            optimizer.zero_grad()
            outputs_train = model(inputs_train)
            loss = loss_fn(outputs_train, labels_train)
            loss.backward()
            optimizer.step()

            batch_size = inputs_train.size(0)
            running_train_loss += loss.item() * batch_size
            total_samples += batch_size

            _, predicted_train = torch.max(outputs_train, 1)

            all_preds_train.extend(predicted_train.cpu().numpy())
            all_labels_train.extend(labels_train.cpu().numpy())

        train_loss = running_train_loss / total_samples
        train_losses.append(train_loss)

        acc_train = accuracy_score(all_labels_train, all_preds_train)
        train_acc.append(acc_train)

        print(f"Época {epoch + 1}/{num_epoch} - Perda no treinamento: {train_loss:.6f} - Acc: {100 * acc_train:.2f}%")

    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'ro-')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.savefig(f'loss_acc_fold-{fold+1}.png', bbox_inches='tight')

    model.eval()
    with torch.no_grad():
        all_preds_test = []
        all_labels_test = []

        for images_test, labels_test in testloader:
            images_test = images_test.view(-1, 164*164*3).to(device)
            labels_test = labels_test.to(device)       
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test, 1)
            all_preds_test.extend(predicted_test.cpu().numpy())
            all_labels_test.extend(labels_test.cpu().numpy())

        acc_test = accuracy_score(all_labels_test, all_preds_test)
        precision_test = precision_score(all_labels_test, all_preds_test)
        recall_test = recall_score(all_labels_test, all_preds_test)
        f1_test = f1_score(all_labels_test, all_preds_test)
    
    print(f'\nAcurácia para o Fold {fold+1}: {100 * acc_test:.2f}%')
    results_acc[fold] = (100 * acc_test)

    print(f'Precisão para o Fold {fold+1}: {100 * precision_test:.2f}%')
    results_precision[fold] = (100 * precision_test)

    print(f'Recall para o Fold {fold+1}: {100 * recall_test:.2f}%')
    results_recall[fold] = (100 * recall_test)

    print(f'F1 para o Fold {fold+1}: {100 * f1_test:.2f}%')
    results_f1[fold] = (100 * f1_test)

    cm = confusion_matrix(all_labels_test, all_preds_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inliers', 'Outliers'])
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Rótulo previsto')
    plt.ylabel('Rótulo verdadeiro')
    plt.savefig(f'matrizconfusao_fold-{fold+1}.png', bbox_inches='tight')

    print("\n!!!Teste finalizado!!!")

training_time = time.time() - training_start_time
print(f"\nTempo total de treinamento: {training_time:.2f} segundos")

print(f'\nResultados Acc: {results_acc}')
print(f'Resultados Precision: {results_precision}')
print(f'Resultados Recall: {results_recall}')
print(f'Resultados F1: {results_f1}')

print(f'\nMédia Acc: {sum(results_acc.values()) / k_folds}')
print(f'Média Precision: {sum(results_precision.values()) / k_folds}')
print(f'Média Recall: {sum(results_recall.values()) / k_folds}')
print(f'Média F1: {sum(results_f1.values()) / k_folds}')