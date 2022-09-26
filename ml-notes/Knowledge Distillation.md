Knowledge Distillation



1. Method for training a smaller student model to mimic the behavior of a slower, larger, but better-performing teacher.
2. Augment the ground truth labels with a distribution of “soft probabilities” from the teacher which provide complementary information for the student to learn from.
3.  Teacher model output vector is  “soften” the probabilities by scaling the logits with a temperature hyperparameter T before applying the softmax.
4. student also produces softened probabilities.

5. (KL) divergence to measure the difference between the two probability distributions.
6. With the KL divergence we can calculate how much is lost when we approximate the probability distribution of the teacher with thestudent. This allows us to define a knowledge distillation loss: L (KD) = T^2D(KL)

T^2 is a normalization factor to account for the fact that the magnitude of the gradients produced by soft labels scales.

For classification tasks, the student loss is  then a weighted average of the distillation loss with the usual cross-entropy loss
L(ce) of the ground truth labels:  L student = αL CE + (1 − α)L KD

α is a hyperparameter that controls the relative strength of each loss.

the temperature is set to 1 at inference time to recover the standard softmax probabilities.







Steps:

1. Finetune the teacher model on the custom dataset.
2. Create a new loss function which is a combination of Cross Entropy +  L(KD)  with a hyperparameter-alpha which considers the soft probabilities  (by scaling the logits with a temparature(T) hyperparameter before applying the softmax) of the teacher model and the student model.
3. Finetune the student model distilbert on the custom dataset with Knowledge Distillation approach. 
4. Apply hyperparameter tuning to find best hyperparameters for alpha,temperature and number of epochs.
5. 