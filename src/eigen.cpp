#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;

pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps){
   Vector newAprox = Vector::Random(X.cols());
   newAprox /= newAprox.norm();
   double eigenvalue;

   for(unsigned i = 0; i < num_iter; ++i){
       Vector oldAprox = newAprox;
       newAprox = X * newAprox;
       newAprox /= newAprox.norm();                    // # <a, b> = |a| |b| cos(angle) ; pero como dividimos por la norma tienen norma 1
       double cos_angle = newAprox.dot(oldAprox);      //cos_angle = np.dot(b, old)
       if((1-eps) < cos_angle && cos_angle <= 1){      //(1 - eps) < cos_angle <= 1:
           cout << "\nParé en la iteración {"<<i+1<<"}\n";
           break;
       }
   }
   eigenvalue = newAprox.transpose().dot(X * newAprox); // = lamda * vt * v
   return make_pair(eigenvalue, newAprox);  
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon){
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);
    for(unsigned i = 0; i < num; ++i){
        pair<double, Vector> temp = power_iteration(A, num_iter, epsilon);
        eigvalues(i) = temp.first;
        eigvectors.col(i) = temp.second;
        A = A - temp.first * temp.second * temp.second.transpose();
        cout<<"\nAutovalor "<< i+1 << ": " << temp.first << "\nAutovector " << i+1 << ": [" << temp.second << "]\n\n";
       
    }  
    return make_pair(eigvalues, eigvectors);
}
