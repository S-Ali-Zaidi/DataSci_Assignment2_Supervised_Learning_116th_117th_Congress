## Expanding the Scope: Supervised Classification of Legislative Texts from the 116th and 117th Congresses  

### Motivation  

[Building upon the work in Assignment 1](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main), this project aimed to address the limitations of clustering algorithms in predicting congressional committee referrals. While unsupervised K-means clustering offered valuable insights into the thematic structure of bills, it ultimately failed to align clusters with committee jurisdictions in a consistent and actionable manner.  

To improve upon these results, I expanded the dataset to include **all bills introduced in both the 116th and 117th Congresses**, resulting in a comprehensive collection of over **20,000 bills**. Each bill was labeled with its assigned committee, to use as the ground truth for training multiple **supervised classification models**. The goal was to compare the performance of four distinct machine learning models—Random Forest, XGBoost, Fully Connected Neural Network (FCNN), and Transformer Classifier—in predicting committee assignments.  

This approach incorporated lessons learned from the [first project](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main), and opens up further avenues for exploration and development. 

---
### Process  

Following the **CRISP-DM framework**, I structured the process in to the following phases:  

#### 1. Domain Understanding  

The primary objective was to evaluate the ability of supervised learning models to accurately predict committee assignments based on the semantic content of legislative texts. With an expanded dataset to give sufficient data to put aside for validations and testing spilts, my hope was that supervised learning would outperform clustering by using committee labels as ground truth during training.  

[As noted in my previous report,](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main) with my background having worked as a Staffer in the House of Representatives, I'm familiar with the high volume of legislation passing through Congress regularly, and the critical role of committee assignments in the legislative process.One day, working models could have real-world implications, as machine learning and NLP systems could assist the House Parliamentarian’s office by expediting referrals for less contentious or routine bills, streamlining the legislative process.  

#### 2. Data Understanding  

**Data Collection:**  
- **116th and 117th Congress Bills:** Texts and metadata for over 20,000 House bills were collected from LegiScan and Congress.gov.  
- **Committee Jurisdiction Descriptions:** Updated and refined committee texts were included to serve as contextual information during modeling.  

**Key Additions:**  
- The inclusion of 116th Congress bills expanded the dataset's diversity and thematic coverage.  
- Committee assignments were used as categorical labels for supervised learning.  

**Embeddings:**  
Using the **OpenAI `text-embedding-3-large` model**, each bill was embedded as a **3072-dimensional vector**. These embeddings were pre-normalized by the OpenAI API and served as input features for all models.  

#### 3. Data Preparation  

The data preparation process included:  
- **Preprocessing:** Ensuring uniformity across all bill texts through the same cleaning and normalization pipeline as in Assignment 1.  
- **Handling Long Texts:** Chunking bills exceeding the token limit (8,190 tokens) into smaller sections, embedding them individually, and averaging their embeddings.  
- **Dataset Splits:** Dividing the dataset into training, validation, and test sets with a **70:15:15 split**.  

#### 4. Modeling  

Four models were implemented and compared:  

1. **Random Forest Classifier**  
   - Chosen for its interpretability and robustness to overfitting in high-dimensional spaces.  
   - Used **100 estimators** with **no maximum depth**.  

2. **XGBoost Classifier**  
   - Known for its gradient-boosting framework and ability to capture complex patterns.  
   - Used **100 estimators** with a **maximum depth of 6**, all else set to defaults.  

3. **Fully Connected Neural Network (FCNN)**  
   - Designed with **two hidden layers** of sizes [512, 128], employing dropout and batch normalization for regularization.  
   - Trained using the Adam optimizer for **50 epochs** with a batch size of 128.  

4. **Transformer Classifier**  
   - Adapted to treat each bill embedding as 16 "tokens," enabling attention mechanisms to capture relationships across sections.  
   - Used **2 transformer layers**, **2 attention heads**, and a batch size of 128, trained for 50 epochs.  

#### 5. Evaluation  

Model performance was evaluated on the test set using **accuracy**, **precision**, **recall**, and **F1-score** as metrics. Overall metrics and F1-scores for individual committees are listed below

---
### Results  

The results demonstrated clear advantages of supervised learning over unsupervised clustering. Both neural network architectures (FCNN and Transformer) significantly outperformed Random Forest and XGBoost in overall metrics and committee-specific predictions.  

#### General Results:

| **Metric**                 | **Random Forest** | **XGBoost** | **FCNN** | **Transformer** |
| -------------------------- | ----------------- | ----------- | -------- | --------------- |
| **Overall Accuracy**       | 0.7505            | 0.8585      | 0.92     | 0.90            |
| **Macro Avg Precision**    | 0.81              | 0.88        | 0.92     | 0.90            |
| **Macro Avg Recall**       | 0.51              | 0.76        | 0.89     | 0.89            |
| **Macro Avg F1-Score**     | 0.56              | 0.80        | 0.90     | 0.89            |
| **Weighted Avg Precision** | 0.79              | 0.86        | 0.92     | 0.91            |
| **Weighted Avg Recall**    | 0.75              | 0.86        | 0.92     | 0.90            |
| **Weighted Avg F1-Score**  | 0.74              | 0.86        | 0.92     | 0.90            |

#### F1-Score Breakdown by Model:

| **Committee**                     | **Random Forest** | **XGBoost** | **FCNN** | **Transformer** |
| --------------------------------- | ----------------- | ----------- | -------- | --------------- |
| Administration                    | 0.57              | 0.81        | 0.88     | 0.85            |
| Agriculture                       | 0.46              | 0.81        | 0.87     | 0.87            |
| Appropriations                    | 0.23              | 0.61        | 0.85     | 0.78            |
| Armed Services                    | 0.69              | 0.85        | 0.92     | 0.91            |
| Budget                            | 0.57              | 0.67        | 0.78     | 0.95            |
| Education and Labor               | 0.79              | 0.88        | 0.92     | 0.92            |
| Energy and Commerce               | 0.74              | 0.86        | 0.92     | 0.91            |
| Ethics                            | 0.00              | 0.75        | 1.00     | 1.00            |
| Financial Services                | 0.78              | 0.87        | 0.92     | 0.91            |
| Foreign Affairs                   | 0.83              | 0.91        | 0.95     | 0.93            |
| Homeland Security                 | 0.33              | 0.78        | 0.81     | 0.84            |
| Intelligence                      | 0.00              | 0.55        | 0.93     | 0.88            |
| Judiciary                         | 0.76              | 0.86        | 0.92     | 0.91            |
| Natural Resources                 | 0.83              | 0.87        | 0.94     | 0.93            |
| Oversight and Reform              | 0.68              | 0.82        | 0.87     | 0.85            |
| Rules                             | 0.19              | 0.75        | 0.91     | 0.91            |
| Science, Space, and Technology    | 0.28              | 0.64        | 0.87     | 0.78            |
| Small Business                    | 0.61              | 0.88        | 0.90     | 0.90            |
| Transportation and Infrastructure | 0.74              | 0.83        | 0.91     | 0.88            |
| Veterans' Affairs                 | 0.91              | 0.94        | 0.98     | 0.96            |
| Ways and Means                    | 0.84              | 0.90        | 0.94     | 0.92            |

#### Key Findings:  

1. **Overall Performance:**  
   - To my surprise, the **FCNN** achieved the highest accuracy (92%) and macro F1-score (0.90), followed closely by the **Transformer Classifier**.  
   - While XGBoost provided competitive results, it struggled with imbalanced committee representation that were relatively sparse in the training data, compared to neural networks.  

2. **Committee-Specific Trends:**  
   - As i expected, FCNN and Transformer excelled in predicting committees with large and diverse datasets, such as **Energy and Commerce** and **Judiciary**.  
   - Less represented committees (e.g., **Ethics**, **Intelligence**, which produce less legislation) posed challenges for Random Forest and XGBoost, where FCNN and Transformer still managed to achieved perfect or near-perfect scores. 

3. **Model Efficiency:**  
   - The **Transformer Classifier** was more memory-efficient than the FCNN, taking up less than half the size -- highlighting its scalability for larger datasets and further training. 

#### Challenges:
Beyond the general challenge of creating and cleaning the datasets, which have been detailed in my first project on this topic, some challenges and difficulties that emerged in this process include:
  
- **Smaller Committees:** Imbalances in committee representation remain a challenge, even for advanced models.  
- **Computation Time:** Neural networks required significantly longer training times compared to tree-based models.  
