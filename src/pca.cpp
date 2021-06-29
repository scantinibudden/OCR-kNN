#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components){
  this->alpha = n_components;
}

void PCA::fit(Matrix X){
  //matriz de covarianza:
  Vector v(X.cols());

  for(int k = 0 ; k<X.cols();k++)
	v(k) = X.col(k).mean();

  for(int i = 0; i < X.rows(); ++i)
  	X.row(i) = (X.row(i) - v)/(sqrt(X.rows()-1));

  X = X.transpose() * X;
  this->eigenVectors = get_first_eigenvalues(X, this->alpha, 5000, 1e-16).second;
}

MatrixXd PCA::transform(Matrix X){
  return X*this->eigenVectors;
}


