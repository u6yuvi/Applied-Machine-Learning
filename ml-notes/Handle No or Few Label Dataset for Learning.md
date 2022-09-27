1. Create a baseline using naive Bayes Classifier

2. Working with No Labeled Data

   1. Zero Shot Classification

   Make use of a pretrained model without any additional fine-tuning on your task-specific corpus.

   Using Prompt 

   		1.  Use Masked language model for classification, which uses the masked language model to predict the content of the masked tokens.
   		1. Concat text with the prompt and a mask word.
   		1. Returns the mos likely tokens to fill in the masked spot or query the pipeline for the probability of a few given tokens.

   Using Prompt with Models trained on Text Entailment tasks.

   1. Treat the text we wish to classify as the premise, and then formulate the hypothesis as:

      “This example is about {label}.”
       where we insert the class name for the label.

   2. The entailment score then tells us how likely that premise is to be about that topic, and we can run this for any number of classes sequentially.

   3. Downside 

      1. We need to execute a forward pass for each class, which makes it less efficient than a standard classifier.
      2. Choice of label names can have a large impact on the accuracy, and choosing labels with semantic meaning is generally the best approach

   4. Metrics 

      1. Define a threshold and select all labels above the threshold. 
      2. Pick the top k labels with the k highest scores.
      3. Plot macro and micro f1 score.

   5. Ways to Imporve Zero Shot Pipelines

      1. The way the pipeline works makes it very sensitive to the names of the labels. If the names don’t make much sense or are not easily connected to the texts, the pipeline will likely perform poorly. Either try using different names or use several names in parallel and aggregate them in an extra step.
      2. Another thing you can improve is the form of the hypothesis. By
          default it is hypothesis="This is example is about {}", but you can pass any other text to the pipeline. Depending on the use case, this might improve the performance.

3. Working with Few labels

   1. Apply Data Augmentation 
      1. Back translation 
      2. Token perturbation
   2. Using Embedding as a Lookup Table
      1. Labeled data is embedded with a model and stored with the labels.
      2. When a new text needs to be classified it is embedded as well, and the label is given based on the labels of the nearest neighbors. It is important to calibrate the number of neighbors to be searched for, as too few might be noisy and too many might mix in neighboring groups.
      3. No model fine tuning is required ,ensure the  select an appropriate model that is ideally pretrained on a similar domain to your dataset.Good example could be GPT-2
      4. Embeddings will be at a token level,Aggrtegate it using mean pooling .Ensure Padding token is not considered in the averaging.Use Attention Mask to handle that.
      5. Use Faiss Index to do nearest neigbour lookup.
   3. Finetuning the Model
   4. In-Context and Few Shot learning with Prompts
      1.  create examples of the prompts and desired predictions and continue training the language model on these examples. A novel method called ADAPET uses such an approach and beats GPT-3 on a wide variety of tasks.
   5. Leveraging Unlabeled Data
      1. Domain Adaptation -Continue training it on data from our domain