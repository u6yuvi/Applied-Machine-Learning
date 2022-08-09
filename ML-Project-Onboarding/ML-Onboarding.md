

**ML Project Onboarding Questionnaire**



| Stages                                                | Sub-Stages                                                   | Questions                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify and establish the analytical problem                |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify the data sources available for building a solution  |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Idenitfy a business metric to measure the business value     |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify a data metric to measure performance of the solution |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify performance contraints -- time latency , accuracy etc |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify user experimence with the solution                  |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify personalization requirements if any required with the solution |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify any user input/feedback for the system to learn and adapt |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Assessment on impact of model staleness to decide how often to perform the updates |
| ML-Business Understanding/Data Requirement Discussion |                                                              | Identify any prior development work required before the start of the project |
| ML-Data Pipeline                                      | Data Collection and Description                              | Listing all the data sources available for use               |
| ML-Data Pipeline                                      | Data Collection and Description                              | Identify methods to acquire the data from data sources       |
| ML-Data Pipeline                                      | Data Collection and Description                              | Describe different data format available to process          |
| ML-Data Pipeline                                      | Data Collection and Description                              | Description on the Data Quantity[How much data will be needed?] |
| ML-Data Pipeline                                      | Data Collection and Description                              | Describe any new discovered surface feature                  |
| ML-Data Pipeline                                      | Data Definition and Understanding                            | Data definition questions 1. What is the input x? 2. What features need to be included 3. What is the target y? 4. Define Ground Truth 5. How can we ensure labelers give consistent labels |
| ML-Data Pipeline                                      | Data Definition and Understanding                            | Validate quality of the data 1. Covers important cases(good coverage of input x) 2. Is defined consistently (definition of label y is unambiguous) 3. Has timely feedback from production data (distribution covers data drift and concept drift) 4. Is sized appropriately 5. Is Data augmentation more likely to be helpful |
| ML-Data Pipeline                                      | Data Definition and Understanding                            | Data Exploration and Statistical Analysis/POC-Heuristic      |
| ML-Data Pipeline                                      | Data Definition and Understanding                            | Revisit the project plan and ensure data is sufficient for the task |
| ML-Data Pipeline                                      | Data Tagging                                                 | Ensure ground truth is  well defined                         |
| ML-Data Pipeline                                      | Data Tagging                                                 | Identify the number of documents to be tagged                |
| ML-Data Pipeline                                      | Data Tagging                                                 | Identify right tool for tagging                              |
| ML-Data Pipeline                                      | Data Tagging                                                 | Identify the quantum of work and the number of annotators required [How expensive is data labeling?] |
| ML-Data Pipeline                                      |                                                              | Idenitfy heuristic/weakly supervised/unsupervised technique,if any, to create annotated data |
| ML-Model Pipeline                                     | Data Preparation                                             | Identify and Define Data Cleaning activity ,if any           |
| ML-Model Pipeline                                     | Data Preparation                                             | Identify and Define Data Augmentation activity, if any Checklist: Does augmented data seem realistic? Is the X->y mapping clear? (eg:Can humans recognise it?) Is the algorithm currently doing poorly on it? |
| ML-Model Pipeline                                     | Data Preparation                                             | Identify and Define Feature Engineering activity, if any     |
| Modelling                                             | Model exploration/Retraining                                 | Describe the intended plan for training, testing, and evaluating the model |
|                                                       | Model exploration/Retraining                                 | Look for availability of good published work about similar problems Has the problem been reduced to practice? Is there sufficient literature on the problem?.Literature search for available solutions/open source project Are there pre-trained models we can leverage? |
| Modelling                                             | Model exploration/Retraining                                 | Establish performance baselines on your problem. 1. Random baseline: if your model just predicts everything at random, what's the expected performance? 2. Human baseline: Identify Human Level Performance [HLP].how well would humans perform on this task? 3. Simple heuristic |
| Modelling                                             | Model exploration/Retraining                                 | Description on model specific assumptions about the data     |
| Modelling                                             | Model exploration/Retraining                                 | Description on the resulting models, report on the interpretation of the models if any |
| Modelling                                             | Model exploration/Retraining                                 | Listing the qualities of your generated models (e.g.in terms of accuracy) and rank their quality in relation to each other. Understand how model performance scales with more data |
| Modelling                                             | Model refinement                                             | Perform model-specific optimizations                         |
| Modelling                                             | Model refinement                                             | Perform error analysis to uncover common failure modes       |
| Modelling                                             | Model Testing and Evaluation                                 | Evaluate model on test distribution; understand differences between train and test set distributions  Perform targeted collection of data to address current failure modes |
| Modelling                                             | Model Auditing                                               | Check for accuracy,fairness/bias and other problems: 1. Brainstorm the ways,the system might go wrong:      a. Performance on subsets of data      b. How common are certain errors[eg.FP,FN]      c. Performance on rare classes 2. Revisit model evaluation metric; ensure that this metric drives desirable downstream user behavior 3. Use metrics to assess performance against the identified issues on appropriate slices of data 3. Get business/product owner buy in |
| ML-Integration                                        | Optimisation                                                 | Refactoring of code to optimise Data and Model Pipeline      |
| ML-Integration                                        | Model Endpoints                                              | Identify and build api signatures to integrate into the Product |
| ML-Integration                                        | Test Suite                                                   | Write tests for: Input data pipeline Model inference functionality Model inference performance on validation data Explicit scenarios expected in production (model is evaluated on a curated set of observations) |
| ML-Integration                                        | Documentation                                                | Integrate Module level documentation and add the Model Card  |
| ML-Deployment                                         |                                                              | To be updated------                                          |
| ML-Monitoring                                         |                                                              | Creation of the Metric Dashboard for the live model monitoring |
|                                                       | Integration of the Dashboard with the Module Monitoring Services |                                                              |
|                                                       | Identify and build pipeline for persisting data for model evaluation or retraining |                                                              |







**Gantt Chart to set deliverable timelines.**

![image-20220809232404751](/home/uv/.config/Typora/typora-user-images/image-20220809232404751.png)





[Refer the link for the updated sheet](https://docs.google.com/spreadsheets/d/1jJLCbOHl6joBpvfRjaQUWS_DUUoG1vInC4DYnH4abds/edit?usp=sharing)