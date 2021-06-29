#include <algorithm>
#include <chrono>
#include <iostream>
#include "knn.h"
#include <numeric>

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors){
    // esta funcion guarda en la clase KNNClassifier la cantidad k de vecinos a tomar en cuenta
    this -> kVecinos = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Vector Y){ // Matriz de train sin etiqueta = X ; Etiquetas = Y
    // esta funcion guarda en nuestra clase KNNClassifier la matriz y el vector generados a partir de los dos archivos de datos
    this -> D = X;
    this -> values = Y;
}


Vector KNNClassifier::predict(Matrix test){ // test tiene todas las imagenes que vamos a testear
    //Esta funcion recibe la matriz de test y se encarga de comparar todas las distancias 
    
    auto ret = Vector(test.rows()); // ret[i] = resultados de test[i]
    Matrix D = this -> D;

    for (unsigned k = 0; k < test.rows(); ++k){ // seleccionamos cada imagen
        Vector imagenActual = test.row(k);
        Vector Dnormas(D.rows());

        for(unsigned i = 0; i < Dnormas.rows(); ++i) // iteramos tantas veces como Dnormas.rows
            Dnormas(i) = (D.row(i)-imagenActual.transpose()).squaredNorm(); // creamos vector de distancias.

        // Ordeno indices por distancia.
        vector<int> indice(Dnormas.size());
        iota(indice.begin(), indice.end(), 0);
        sort(indice.begin(), indice.end(), [&Dnormas](size_t a, size_t b) {return Dnormas(a) < Dnormas(b);});
        indice.resize(this->kVecinos);  //me quedo solo con los mas cercanos
        
        // Lleno el vector res con los tags pertinentes a los k cercanos.
        
        vector<int> rta (indice.size());
        for (long unsigned int i = 0; i < indice.size(); i++)
            rta[i] = this->values(indice[i]);

        //rta con los valores ordenados

        vector<int> apariciones(10, 0); // creamos contador de k

        for(long unsigned int i = 0 ; i < rta.size() ; ++i)
            apariciones[rta[i]]++;

        int valor = 0;
        for(long unsigned int i = 0 ; i < apariciones.size() ; ++i)
            if(apariciones[i] > valor)
                valor = i;
    
        ret(k) = valor; // guardamos el resultado
    }

    return ret;
}
