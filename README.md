# DataSwift
This framework uses Inductive Matrix Completion (IMC) with an XGBoost Residual Model for Learned Query Optimization. This model needs only <5 minutes of training to generally surpasses PostgreSQL in directly latency comparison tests. 

A high-level overview of the different componenets of the system are outlined in the image below:


<img width="435" alt="Screenshot 1446-07-27 at 10 02 24 PM" src="https://github.com/user-attachments/assets/ae6189d1-40df-4ca4-bdd0-defdfcabc2a9" />

The code can automatically connect to your database and select the best configuration parameters to minimize latency. Please try the model out!

HOW TO USE:

Download U_best, V_best, Y_scaled, and the feature_scaler. After, execute train9v3test1 to test the model out!


More Details Regarding Contribution + Architecture Coming Soon!
