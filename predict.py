from train import models
import cleaning as clean
#just checking if it works
df = clean.get_clean_data()
df =df.head(1)
features,_ = df.drop(columns=['MAX_SEVERITY_LEVEL']), df.MAX_SEVERITY_LEVEL
for model in models:
    print(model.predict(features))