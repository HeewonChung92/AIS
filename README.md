# AIS (Abbreviate Injury Score)
Paper: Artificial Intelligence to Predict In-Hospital Mortality Using Anatomical Injury Score


##  Abstract  

The aim of the study is to develop an artificial intelligence (AI) algorithm based on a deep learning model to predict the mortality using abbreviate injury score (AIS). The performance of the conventional anatomic injury severity score (ISS) system in predicting the in-hospital mortality is still limited. AIS data of 42,933 patients registered in the Korean trauma data bank from four Korean regional trauma centers were initially considered. After filtering out the patients aged less than 19 or those who spent less than 6 hours in the hospital, we considered 37,762 patients, of which 36,493 (96.6%) survived and 1,269 (3.4%) deceased. To enhance the AI model performance, we reduced the AIS codes to 46 input values by organizing them according to the organ location (Region-46). The total AIS and six categories of the anatomic region in the ISS system (Region-6) were used to compare the input features. The AI models were compared with the conventional ISS and new ISS (NISS) systems. We evaluated the performance pertaining to the 12 combinations of the features and models. The highest accuracy (85.05%) corresponded to Region-46 with DNN, followed by that of Region-6 with DNN (83.62%), AIS with DNN (81.27%), ISS-16 (80.50%), NISS-16 (79.18%), NISS-25 (77.09%), and ISS-25 (70.82%). The highest AUROC (0.9084) corresponded to Region-46 with DNN, followed by that of Region-6 with DNN (0.9013), AIS with DNN (0.8819), ISS (0.8709), and NISS (0.8681). The proposed deep learning scheme with feature combination exhibited high accuracy metrics such as the balanced accuracy and AUROC than the conventional ISS and NISS systems. We expect that our trial would be a cornerstone of more complex combination model. 

##  Requirement  
This code was  written in python 3.6 and Tensorflow 2.0. 
