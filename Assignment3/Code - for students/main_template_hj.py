# %% imports
# libraries
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
# local imports
import MNIST_dataloader
import autoencoder_template
from bokeh.plotting import figure
from bokeh.io import show
import numpy as np
import matplotlib.cm
# set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D:\\gittutorial-TootJack\\5LSL0_MSPD\\Code - for students' #change the data location to something that works for you
batch_size = 64
no_epochs = 500
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
AE = autoencoder_template.AE()
AE.cuda() # Transfer to GPU

# create the optimizer
optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

## %% training loop
# go over all epochs
# loss_list = []
# for epoch in range(no_epochs):
#     print(f"\nTraining Epoch {epoch+1}:")
#     loss_sum = 0
#     # go over all minibatches
#     for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
#         # fill in how to train your network using only the clean images
#         AE.train()
#         optimizer.zero_grad()
#         x_clean = x_clean.cuda() # Transfer to GPU
#         # Run the forward pass
#         output,feature = AE(x_clean)
#         loss = F.mse_loss(output, x_clean)

#         # Backprop
#         loss.backward()

#         # Perform the Adam optimizer
#         optimizer.step()
#         loss_sum += loss.item()
#     loss_list.append(loss_sum/len(test_loader.dataset.Clean_Images))
# # save the trained model
# torch.save(AE.state_dict(), data_loc+'/Model/state.pt')

# # %%
# # Plot the loss 
# p = figure(x_axis_label='Number of epochs',y_axis_label='Train Loss', width=850,x_range = (0,no_epochs), title='Train loss')
# p.line(np.arange(len(loss_list)), loss_list, legend_label="Train Loss")
# show(p)

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]
# load the trained model
AE.load_state_dict(torch.load(data_loc+'/Model/state.pt'))
AE.eval()
AE = AE.cpu()
x_output_example,x_feature_example = AE(x_clean_example)
# show the examples in a plot
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
        
    plt.subplot(2,10,i+11)
    plt.imshow(x_output_example[i,0,:,:].detach().numpy(),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.tight_layout()
plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
plt.show()


# %%# Combine digits to track as a tensor
numbers = []
for x in range(10):
    index = (labels_test == x).nonzero().reshape(-1)
    cache = torch.index_select(x_clean_test, 0, index)
    _,vectors = AE(cache)
    numbers.append(vectors)

# %%# Visulize the latent space
colors = matplotlib.cm.Paired(np.linspace(0, 1, len(numbers)))
fig, ax = plt.subplots()
ax.set_title('Latent space')
ax.set_xlabel('x1 (node 1)')
ax.set_ylabel('x2 (node 2)')
for (points, color, digit) in zip(numbers, colors, range(10)):
    ax.scatter([item[0][0][0] for item in points.detach().numpy()],
                [item[0][0][1] for item in points.detach().numpy()], color=color, label='digit    {}'.format(digit))
    ax.grid(True)
ax.legend()

# %%#  Use the latent space to do some rudimentary classification
from sklearn.neighbors import NearestNeighbors

# Get train features
model = AE
_,train_feature = model(x_clean_train)
train_feature = train_feature.detach().numpy()
train_features = []
for features in train_feature:
    train_features.append(features[0][0])
# Closest Euclidean distance -> KNN
knn_gallery = NearestNeighbors(n_neighbors=1).fit(train_features)
test_features = []
for i, points in zip(range(10),numbers):
    items = []
    print('Length of number',i,'in test set is', len(points.detach().numpy()))
    for item in points.detach().numpy():
        items.append(item[0][0])
    test_features.append(items)
for i in range(10):
    predictions = knn_gallery.kneighbors(test_features[i], return_distance=False)
    prediction_tensor = torch.LongTensor(predictions).squeeze()
    # Get the predicted label
    label_pred = torch.index_select(labels_train, 0, prediction_tensor)
    # Index of corectly labeled images
    index = (label_pred == i).nonzero().reshape(-1)
    accuracy = 100*len(index)/len(test_features[i])
    print('Accuracy of number',i,'is', accuracy,'%')

# %%# Define Average_Precision algorithm

# %%# Make initial embedding space, th
# %%# Make initial embedding space, th

