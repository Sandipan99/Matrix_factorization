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
			A(i,j) = (float)(rand()%a) + 1;
	}
	return A;
}

arma::mat fill_random_norm(arma::mat A,int a){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = (float)(rand())/RAND_MAX;
	}
	return A;
}

arma::mat fill_zero(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = 0.0;
	}
	return A;
}

arma::vec return_row(arma::mat A,int index){
	arma::vec B(A.n_cols);
	for(int i=0;i<A.n_cols;i++)
		B(i) = A(index,i);
	return B;
}

arma::vec return_col(arma::mat A,int index){
	arma::vec B(A.n_rows);
	for(int i=0;i<A.n_rows;i++)
		B(i) = A(i,index);
	return B;
}

float dot_product(arma::vec A, arma::vec B, int size){
	float prod = 0.0;
	for(int i=0;i<size;i++)
		prod+=A(i)*B(i);
	return prod;
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



void display(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			std::cout << A(i,j) << " ";
		std::cout << std::endl;
	}
}

void display(arma::vec A,int k){
	for(int i=0;i<k;i++)
		std::cout << A(k,1) << " ";
}

int main(int argc, char *argv[]){
	int user = 5,item =4; 
	int k = 2,steps = 5000;
	float alpha = 0.0002, beta = 0.01;
	arma::mat R = {{5,3,0,1},{4,0,0,1},{1,1,0,5},{1,0,0,4},{0,1,5,4}};
	display(R);
	arma::mat R_bar;
	arma::mat P(5,k);
	arma::mat Q(4,k);
	arma::vec A;
	arma::vec B;
	P = fill_random_norm(P,4);
	Q = fill_random_norm(Q,4);
	R_bar = calculate_error(R,P,Q,alpha,beta,steps,k);	
	display(R_bar);	
	return 0;
}
