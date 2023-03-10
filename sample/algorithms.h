#ifndef _ALGORITHMS_H_
#define _ALGORITHMS_H_

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <map>
#include <new>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <stack>
#include <string>
#include <sstream>
#include <random>
#include <limits>
#include <cassert>
#include <parallel/algorithm>
#include <parallel/numeric>
//#include <parallel/random_shuffle>

#include "CSR.h"

#include "utility.h"

#include "commonutility.h"


using namespace std;

#define VALUETYPE float
#define SREAL 1 /* needed for intrinsic, use DREAL for double */
#define INDEXTYPE unsigned int
#define MAXBOUND 5
#define MAXMIN 3.0
#define SM_TABLE_SIZE 2048
#define SM_BOUND 6.0
#define t 0.99
#define PI 3.14159265358979323846
static int PROGRESS = 0;

const VALUETYPE SM_RESOLUTION = SM_TABLE_SIZE / (2.0 * SM_BOUND);

class algorithms{
	public:
		CSR<INDEXTYPE, VALUETYPE> graph;
		VALUETYPE *nCoordinates;
		VALUETYPE GAMMA = 1.0;
		INDEXTYPE DIM;
		string filename;
		string outputdir;
		int rank;
	public:
	algorithms(string input, string outputd, INDEXTYPE dim, VALUETYPE gm, INDEXTYPE bsize, int rank) {
		outputdir = outputd;
		filename = input;
		DIM = dim;
		GAMMA = gm;
		this->rank = rank;
		//printCSR(A_csr);
		//nCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows * dim])));
	}
	~algorithms(){
		if(rank==0)
			free(nCoordinates);
	}

	/********************/

	void initInputFile(CSR<INDEXTYPE, VALUETYPE> &A_csr) {
		graph.make_empty();
		graph = A_csr;
		nCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows * DIM])));	
	}

	void randInitF();
	void randInit();
	VALUETYPE scale(VALUETYPE v);

	// Force2Vec with Fruchterman Reingold 
	// grad f_a = -||z_u - z_v ||^2
	// grad f_r = 1 / ||z_u - z_w ||
	vector<VALUETYPE> AlgoForce2VecFR(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with Force Atlas 2
	// grad f_a = -||z_u - z_v ||
	// grad f_r = 1 / ||z_u - z_w ||
	vector<VALUETYPE> AlgoForce2VecFA(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	
	// Force2Vec with LinLog
	// grad f_a = -log(1 + ||z_u - z_v ||)
	// grad f_r = 1 / ||z_u - z_w ||
	vector<VALUETYPE> AlgoForce2VecLL(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	
	// Force2Vec without heuristic (attractive: neighbors, repulsive: others)
	// Also uses loglike with EPS and stepsize gets smaller by mult. factor each round. 
	// Runs for a given amount of loops even if not part of the signature.
	// ToDo: Check if 100% correct for attractive/repulsive calculation
	vector<VALUETYPE> AlgoForce2Vec(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);	
	
	// Force2Vec without heuristic (attractive: neighbors, repulsive: others)
	// Stepsize gets smaller substracting fraction of finished loops.
	// Runs for a given amount of loops even if not part of the signature.
	// NOT USED IN CODE!
	vector<VALUETYPE> AlgoForce2VecBR(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	
	// Force2Vec with t-distribution, direct neighborhood and negative sampling (same)
	// Stepsize is static and runs for given amount of loops
	// ToDo: Unclear whether there is an error on line 610: forceDiff might be negative
	// Also possible opt.: multiplication with the denominator only has to be once for the sum
	vector<VALUETYPE> AlgoForce2VecNS(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with t-distribution, direct neighborhood and negative sampling (different for each batch)
	// Stepsize is static and runs for given amount of loops
	// ToDo: Unclear whether there is an error on line 715 similar to 610
	// Also possible opt.: multiplication with the denominator only has to be once for the sum
	//vector<VALUETYPE> AlgoForce2VecNSBS(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with t-distribution, direct neighborhood and negative sampling (different for each batch)
	// Currently reusing code (and overwriting) from AlgoForce2VecNSBS. Note: this does use only ns per batch as in AlgoForce2VecNS
	// Implements Distributed Batch-split implementation
	vector<VALUETYPE> AlgoForce2VecNSD(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with sigmoid, direct neighborhood and negative sampling (same)
	// Stepsize is static and runs for given amount of loops
	// Computation of gradients is already well optimized
	vector<VALUETYPE> AlgoForce2VecNSRW(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with sigmoid, direct neighborhood and negative sampling (different for each batch)
	// Stepsize is static and runs for given amount of loops
	// Computation of gradients is already well optimized
	vector<VALUETYPE> AlgoForce2VecNSRWBS(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	// Force2Vec with sigmoid, random walk neighborhood and negative sampling (same)
	// Stepsize is static and runs for given amount of loops
	vector<VALUETYPE> AlgoForce2VecNSRWEFF(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	
	// Force2Vec with sigmoid, random walk neighborhood and negative sampling (different for each batch)
	// Stepsize is static and runs for given amount of loops
	// NOT USED IN CODE!
	vector<VALUETYPE> AlgoForce2VecNSRWEFFBS(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
#ifdef AVX512
	vector<VALUETYPE> AlgoForce2VecNS_SREAL_D128_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	
	vector<VALUETYPE> AlgoForce2VecNSRW_SREAL_D128_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	
	vector<VALUETYPE> AlgoForce2VecNSRWEFF_SREAL_D128_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	vector<VALUETYPE> AlgoForce2VecNSRWEFF_SREAL_D64_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);

	vector<VALUETYPE> AlgoForce2VecNSLB_SREAL_D128_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	vector<VALUETYPE> AlgoForce2VecNSLB_SREAL_D64_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr);
	vector<VALUETYPE> AlgoForce2VecNSRWLB_SREAL_D64_AVXZ(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr); 
#endif	

/********************/

	void print(){
		for(INDEXTYPE i = 0; i < graph.rows; i += 1){
			int ind = i * DIM;
			cout << "Node:" << i << ":";
			for(INDEXTYPE d = 0; d < DIM; d++){
                		cout << nCoordinates[ind+d] << " ";
        		}
			cout << endl;
		}
		cout << endl;
	}
	void writeToFile(string f){
		stringstream  data(filename);
    		string lasttok;
    		while(getline(data,lasttok,'/'));
		filename = outputdir + lasttok + f + ".embd";
		ofstream output;
		output.open(filename);
		cout << "Creating output file in following directory:" << filename << endl;
		output << graph.rows << " " << DIM << endl;
		for(INDEXTYPE i = 0; i < graph.rows; i += 1){
			output << i + 1 << " ";
			INDEXTYPE iindex = i * DIM;
			for(INDEXTYPE d = 0; d < this->DIM; d++){
				output << nCoordinates[iindex+d] <<" ";
			}
			output << endl;
		}
		output.close();
	}
};
#endif
