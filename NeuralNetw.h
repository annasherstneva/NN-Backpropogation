#pragma once
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

class NeuralNetw
{
public:
	NeuralNetw(int InputN, int OutputN, int HideN,double speed);
	void Train(std::vector<std::vector<double>> TrainDataSet, std::vector <double> Labels);
	double* check_output(double * hide_output);
	~NeuralNetw(void);

private:
	int Input_N;
	int Output_N; 
	int Hide_N;
	double speed_learning;

	double ModCoef;

	double** Hide_Weights;
	double** Output_Weights;

	double* input;
	double* output;
	double* hide_output;
	double* deltaHide;
	double* deltaOutput;
	double* sumHide;
	double* sumOutput;
	double* GradH;
	double* GradO;

	void initialize_weights();
	void init_delta_mas();
	double** memory_weights(int before, int current);
	void nul_mas(int size, double* mas);
	double* softmax(double* X);
	double* check_hide_output(double * input);
	double BinSigmFun(double x);
	void check_grad_out(double *y);
	void Check_grad(double *y);
	void Change_Weights(double * Grad_o, double * Grad_h);
	void Change_Delta(double * Grad_o, double * Grad_h);
	double CrossEntropy(std::vector<std::vector<double>> TrainDataSet, std::vector <double> Labels);
};

