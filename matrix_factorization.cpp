#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<armadillo>
#include<fstream>
#include<math.h>

/**implements simple matrix factorization algorithm
input user-item matrix, alpha, beta(regularizer), k(latent factors)
@author Sandipan Sikdar
**/

arma::mat fill_random(arma::mat A,int a){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = (float)(rand()%a + 1);
	}
	return A;
}


arma::mat fill_random(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = 0.0;
	}
	return A;
}

arma::mat calculate_error(arma::mat R,arma::mat P,arma::mat Q,float alpha,float beta,int steps,int d){
	Q = Q.t();
	arma::mat R_bar;
	arma::vec P_i(P.n_cols);
	arma::vec Q_j(Q.n_rows);
	float e_i_j,e;
	int it = steps;
	while(it>0){
		for(int i=0;i<R.n_rows;i++){
			for(int j=0;j<R.n_cols;j++){
				if(R(i,j)>0){
					P_i = return_row(P,i);
					Q_j = return_col(Q,j);
					e_i_j = R(i,j) - dot_product(P_i,Q_j,d);
					for(int k=0;k<d;k++){
						P(i,k)+= alpha*(2*e_i_j*Q(k,j) - beta*P(i,k));
						Q(k,j)+= alpha*(2*e_i_j*P(i,k) - beta*Q(k,j));	
					}
				}	
			}
		}
		e = 0.0;
		for(int i=0;i<R.n_rows;i++){
			for(int j=0;j<R.n_cols;j++){
				if(R(i,j)>0){
					P_i = return_row(P,i);
					Q_j = return_col(Q,j);
					e+= pow(R(i,j)- dot_product(P_i,Q_j,d),2);
					for(int k=0;k<d;k++){
						e+= (beta/2)*(pow(P(i,k),2) + pow(Q(k,j),2));
					}
				}
			}
		}
		if(e<0.001)
			break;
		it--;
	}
	R_bar = P*Q;
	return R_bar;
}

arma::vec return_row(arma::mat A,int index){
	arma::vec B(A.cols);
	for(int i=0;i<A.cols;i++)
		B(i) = A(index,i);
	return B;
}

arma::vec return_col(arma::mat A,int index){
	arma::vec B(A.rows);
	for(int i=0;i<A.rows;i++)
		B(i) = A(i,index);
	return B;
}

float dot_product(arma::vec A, arma::vec B, int size){
	float prod = 0.0;
	for(int i=0;i<size;i++)
		prod+=A(i)*B(i);
	return prod;
}

void display(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			std::cout << A(i,j) << " ";
		std::cout << std::endl;
	}
}

int main(int argc, char *argv[]){
	int user = 5,item =4;
	int k = 2,steps = 5000;
	float alpha = 0.0002, beta = 0.01;
	arma::mat R(5,4);
	arma::mat R_bar(5,4);
	arma::mat P(5,k);
	arma::mat Q(4,k);
	P = fill_random(P,5);
	Q = fill_random(Q,5);
	R_bar = calculate_error(R,R_bar,P,Q,alpha,beta,steps,k);
	display(R_bar);	
	return 0;
}
