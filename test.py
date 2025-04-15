'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from dataset import my_bidict
from tqdm import tqdm
from pprint import pprint
import argparse
from bidict import bidict
import csv
import pandas as pd
NUM_CLASSES = len(my_bidict)
my1_bidict = bidict({'Class0': 0, 
                    'Class1': 1,
                    'Class2': 2,
                    'Class3': 3,
                    'Undefine': -1})

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    # DLML Compares real data/distribution and the synthesized and we will find the loss
    dlml = lambda real_data, gen_data : discretized_mix_logistic_loss(real_data, gen_data)  
    answer = []
    for img in model_input:
        loss_list=[]
        img_batch = img.unsqueeze(0)
        for i in range(NUM_CLASSES):
            label_tensor = torch.tensor([i],dtype=torch.long,device=device)
            model_output = model(img_batch, labels=label_tensor, sample=False)
            loss = dlml(img_batch, model_output)
            loss_list.append(loss.item())
        answer.append(np.argmin(loss_list))
    return torch.tensor(answer).to(device)
# End of your code

# def classifier(model, data_loader, device):
#     model.eval()
#     acc_tracker = ratio_tracker()
#     answers=[]
#     for batch_idx, item in enumerate(tqdm(data_loader)):
#         model_input, categories = item
#         model_input = model_input.to(device)
#         #original_label = [value for item, value in categories]
#         #original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
#         answer=(get_label(model, model_input, device))
        
#         correct_num = torch.sum(answer == 7)
#         acc_tracker.update(correct_num.item(), model_input.shape[0])
#         answers.append(answer)
#     return acc_tracker.get_ratio(),answers
def classifier(model, data_loader, device, mode='test'): # Add 'mode' parameter
    model.eval()
    acc_tracker = ratio_tracker()
    all_predictions = [] # <--- Initialize list to store all predictions

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)

        # Get predictions for the current batch
        answer = get_label(model, model_input, device)

        # Append predictions for this batch to the master list
        # Use .cpu().tolist() to detach from graph, move to CPU, and convert to list
        all_predictions.extend(answer.cpu().tolist()) # <--- Accumulate predictions

        # --- Accuracy Calculation (Only run if NOT in 'test' mode) ---
        if mode != 'test':
            try:
                # Get original labels ONLY if not in test mode
                original_label = [my_bidict[cat] for cat in categories] # Use my_bidict
                original_label = torch.tensor(original_label, dtype=torch.int64).to(device)

                # Calculate correct predictions for the batch
                correct_num = torch.sum(answer == original_label)
                acc_tracker.update(correct_num.item(), model_input.shape[0])
            except KeyError as e:
                # Handle cases where validation/train data might have unexpected categories
                print(f"\nWarning: Encountered invalid category '{e}' during accuracy calculation. Skipping accuracy update for this batch.")
            except Exception as e:
                 print(f"\nWarning: Error during accuracy calculation: {e}. Skipping update.")
        # --- End Accuracy Calculation ---

    # After the loop, convert the list of all predictions to a single tensor
    final_predictions_tensor = torch.tensor(all_predictions) # <--- Create the final tensor

    # Return accuracy (or -1 if not calculated) and the *full* prediction tensor
    accuracy = acc_tracker.get_ratio() if mode != 'test' else -1.0
    return accuracy, final_predictions_tensor # <--- Return all predictions
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    #model = random_classifier(NUM_CLASSES)
    input_channels=3
    model = PixelCNN(nr_resnet = 3, nr_filters = 80, input_channels = 3, nr_logistic_mix = 10, num_classes = 4, embedding_dim = 16)
    
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
        print(f"model we are evaluating on is {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    # Pass the mode to the classifier function
    acc, label_tensor = classifier(model=model, data_loader=dataloader, device=device, mode=args.mode) # <--- Pass args.mode

    # Only print accuracy if it was calculated (i.e., not in 'test' mode)
    if args.mode != 'test':
         print(f"Accuracy: {acc}")
    else:
         print(f"Accuracy: N/A (running in '{args.mode}' mode)") # <--- Indicate accuracy isn't available

    # --- Keep the CSV writing code as is ---
    print("\nAttempting to write predicted labels to data/test.csv...")
    try:
        # labels_array = label_tensor.cpu().numpy() #<-- No need for .cpu() if tensor created from list
        labels_array = label_tensor.numpy() # Convert the full tensor

        csv_path = os.path.join(args.data_dir, 'test.csv')
        df = pd.read_csv(csv_path, header=None)

        if len(labels_array) != len(df):
            raise ValueError(f"Length mismatch: Number of predicted labels ({len(labels_array)}) "
                             f"does not match the number of rows in '{csv_path}' ({len(df)}).")

        df.iloc[:, 1] = labels_array
        df.to_csv(csv_path, index=False, header=False)
        print(f"Successfully updated '{csv_path}' with predicted labels.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except ImportError:
        print("Error: pandas library not found. Make sure it's installed (pip install pandas).")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

        
       
