import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
wine=load_wine()
x=wine.data
y=wine.target

mlflow.set_tracking_uri("http://127.0.0.1:5000") #due to some bug mlflow is unable to log files through log_artifac() so to overcome that this code is used
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

max_depth=5
n_estimators=20

# Mention yout experiment below
mlflow.set_experiment("MLOps adding experiment demo ") # This is an important code in which we can give name for particular experiment for every file, where each file's change will be saved along with its params, artifacts, metrics in separate folder

with mlflow.start_run():
    model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    model.fit(x_train,y_train)
    
    preds=model.predict(x_test)
    accuracy_score=accuracy_score(y_test,preds)
    cm=confusion_matrix(preds,y_test)
    
    #Save Plots
    plt.figure(figsize=(10,7))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("confusion_matrix.png")
    
    #save logs 
    #save metrics
    mlflow.log_metric('accuracy',accuracy_score) #used to save logs for metrics
    
    #save parameters
    mlflow.log_param('max_depth',max_depth) #used for parameters
    mlflow.log_param('n_estimators',n_estimators) #same as above
    
    #save files and artifacts
    mlflow.log_artifact("confusion_matrix.png") # used to log images plots, files etc
    mlflow.log_artifact(__file__) # here the python file itself is logged
    
    #save tags
    mlflow.set_tags({"author":"Bhuvnn","Project":"Wine Classification"})
    
    #log the model
    mlflow.sklearn.log_model(model,"RandomForest Model")
    print(round(accuracy_score*100,2))
    