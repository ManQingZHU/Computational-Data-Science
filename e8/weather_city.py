import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

OUTPUT_TEMPLATE = (
    'scale -> PCA -> SVM:      {svm:.3g} \n'
)

def main():
	data = pd.read_csv(sys.argv[1])
	x = data.iloc[:,1:].values
	y = data['city'].values

	#split training data and testing data
	X_train, X_test, y_train, y_test = train_test_split(x, y)

	std_pca_svc = make_pipeline(
		StandardScaler(),
    	PCA(15),
    	SVC(kernel='linear', C = 4.5) 
	)

	std_pca_svc.fit(X_train, y_train)

	print(OUTPUT_TEMPLATE.format(
		svm= std_pca_svc.score(X_test, y_test),
		))

	data2 = pd.read_csv(sys.argv[2])
	x2 = data2.iloc[:,1:].values 
	predictions = std_pca_svc.predict(x2)
	pd.Series(predictions).to_csv(sys.argv[3], index=False)


if __name__ == '__main__':
    main()



