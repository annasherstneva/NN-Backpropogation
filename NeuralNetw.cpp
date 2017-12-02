#include "NeuralNetw.h"
#include <vector>
#include <iostream>


NeuralNetw::NeuralNetw(int InputN, int OutputN, int HideN, double speed)
{
	Input_N = InputN;
	Output_N = OutputN;
	Hide_N = HideN;

	input = new double [InputN];
	output = new double [OutputN];
	hide_output = new double [HideN];
	deltaHide = new double [HideN];
	deltaOutput = new double [OutputN];
	sumHide = new double [HideN];
	sumOutput = new double [OutputN];
	GradH = new double [HideN];
	GradO = new double [OutputN];

	speed_learning = speed;
	Hide_Weights = memory_weights(InputN, HideN);
	Output_Weights = memory_weights(HideN, OutputN);

	nul_mas(Hide_N, sumHide);
	nul_mas(Output_N, sumOutput);
	nul_mas(Hide_N, GradH);
	nul_mas(Output_N, GradO);

	initialize_weights();
	init_delta_mas();
};

void NeuralNetw::Train (std::vector<std::vector<double>> TrainDataSet, std::vector <double> Labels)
{
	double * T = new double [Output_N];
	

	double sum = 0.0;		
	int epoch = 0;
	while (epoch<40)
	{
		int ok = 0;//number of right answers
		std::cout << "Number of ep = " << epoch << " Computing cross-entropy..." << std::endl;
		double cross = 0.0;
		cross = CrossEntropy(TrainDataSet, Labels);
		std::cout << "err: " << cross << " ";
		std::cout << std::endl;
		if (cross <= 0.1)
		{
			break;
		}
		std::cout << ": Calculating output and changing weights..." << std::endl;
		for (int i = 0; i < TrainDataSet.size(); i++)
		{
			for (int j = 0; j < Input_N; j++)
			{
				input[j] = TrainDataSet[i][j];
			}
			for (int j = 0; j < Output_N; j++)
			{
				T[j] = 0.0;
				if (j == Labels[i])
					T[j] = 1.0;
			}
			
			output = check_output(check_hide_output(input));
			for(int j=0;j< Output_N;j++)
			{
				if (output[j] > 0.9)
					if (T[j] == 1.0)
						ok++;
			}
			

			Check_grad(T);

			Change_Weights(GradO, GradH);
			Change_Delta(GradO, GradH);
		}
		std::cout << "Number of right answers: " << ok << std::endl;
		epoch++;
		if (ok >= TrainDataSet.size()*0.90)
		{
			std::cout << "Number of right answers > 90%" << std::endl;
			break;
		}
	}
};

double* NeuralNetw::check_hide_output(double * input)
{
	nul_mas(Hide_N, sumHide);
	for (int i = 0; i< Hide_N; i++)
	{
		for (int j = 0; j < Input_N; j++)
		{
			sumHide[i] += input[j] * Hide_Weights[j][i];
		}
	}
	for (int i = 0; i < Hide_N; i++)
	{
		sumHide[i] += deltaHide[i];
	}

	for (int i = 0; i < Hide_N; i++)
	{
		hide_output[i] = BinSigmFun(sumHide[i]);
	}
	return hide_output;
};

double* NeuralNetw::check_output(double *hide_output)
{
	nul_mas(Output_N, sumOutput);
	for (int i = 0; i < Output_N; i++)
	{
		for (int j = 0; j < Hide_N; j++)
		{
			sumOutput[i] += hide_output[j]*Output_Weights[j][i];
		}
	}
	for (int i = 0; i < Output_N; i++)
	{
		sumOutput[i] += deltaOutput[i];
	}

	output = softmax(sumOutput);
	return output;	
};

double NeuralNetw::BinSigmFun(double x)
{
		return 1/(1+exp(-x));
};

double** NeuralNetw::memory_weights(int before, int current)
{
	double ** mas;
	mas = new double *[before];
	for (int i=0;i<before;i++)
	{
		mas[i] = new double[current];
	}
	return mas;
};

void NeuralNetw::nul_mas(int size, double* mas)
{
	for (int i=0;i<size;i++)
	{
		mas[i] = 0;	
	}
};

void NeuralNetw:: init_delta_mas()
{
	for (int j = 0; j < Output_N; j++)
	{
		deltaOutput[j] = (double)rand() * (0.5 - (-0.5)) / RAND_MAX + (-0.5);
	}
	for (int j = 0; j < Hide_N; j++)
	{
		deltaHide[j] = (double)rand() * (0.5 - (-0.5)) / RAND_MAX + (-0.5);
	}
};

void NeuralNetw::initialize_weights()
{
	for (int i = 0; i < Input_N; i++)
	{
		for (int j = 0; j < Hide_N; j++)
		{
			Hide_Weights[i][j] = (double)rand() * (0.5 - (-0.5)) / RAND_MAX + (-0.5);
		}
	}
	for (int i = 0; i < Hide_N; i++)
	{
		for (int j = 0; j < Output_N; j++)
		{
			Output_Weights[i][j] = (double)rand() * (0.5 - (-0.5)) / RAND_MAX + (-0.5);
		}
	}
};

double* NeuralNetw::softmax(double* sumOut)
{
	double * z_exp = new double [Output_N];
	double * soft_max = new double [Output_N];
	for (int i=0;i< Output_N; i++)
	{
		z_exp[i] = exp(sumOut[i]);
	}
	double sum=0;
	for (int i=0;i< Output_N; i++)
	{
		sum+=z_exp[i];
	}
	for (int i=0;i< Output_N; i++)
	{
		soft_max[i] = z_exp[i]/sum;
	}
	return soft_max;
};



void NeuralNetw::check_grad_out(double * t)
{
	for (int i = 0; i < Output_N; i++)
	{
		GradO[i] = (t[i] - output[i]);
	}
};

void NeuralNetw::Check_grad( double * T)
{
	check_grad_out(T);

	double sum = 0.0;
	double der = 0.0;
	for (int i = 0; i < Hide_N; i++)
	{
		for (int j = 0; j < Output_N ; j++)
		{
			sum += GradO[j]*Output_Weights[i][j];
		}
		der = hide_output[i] * (1 - hide_output[i]);
		GradH[i] = sum*der;
	}
};

void NeuralNetw::Change_Weights(double * Grad_o, double * Grad_h)
{
	double delta = 0.0;
	for (int i = 0; i < Hide_N; i++)
	{
		for (int j = 0; j < Output_N; j++)
		{
			delta = speed_learning*Grad_o[j]*hide_output[i];
			Output_Weights[i][j] += delta;
		}
	}
	for (int i = 0; i < Input_N; i++)
	{
		for (int j = 0; j < Hide_N; j++)
		{
			delta = speed_learning*Grad_h[j]*input[i];
			Hide_Weights[i][j] += delta;
		}
	}
};

void NeuralNetw::Change_Delta(double * Grad_o, double * Grad_h)
{
	double delta = 0.0;
	
		for (int j = 0; j < Output_N; j++)
		{
			delta = speed_learning*Grad_o[j];
			deltaOutput[j] += delta;
		}
	
	
		for (int j = 0; j < Hide_N; j++)
		{
			delta = speed_learning*Grad_h[j];
			deltaHide[j] += delta;
		}
	
};

double NeuralNetw:: CrossEntropy(std::vector<std::vector<double>> TrainDataSet, std::vector<double> Labels)
{
	double sum = 0.0;

	double *X = new double[Input_N];
	double *Y = new double[Output_N];
	double *T = new double[Output_N];

	for (int i = 0; i < TrainDataSet.size();i++)
	{
		for (int j = 0; j < Input_N; j++)
		{
			X[j] = TrainDataSet[i][j];
		}
		for (int j = 0; j < Output_N; j++)
		{
			T[j] = 0.0;
			if (j == Labels[i])
				T[j] = 1.0;
		}
		Y = check_output(check_hide_output(X));
		for (int j = 0; j < Output_N;j++)
		{
			sum +=  log(Y[j]) * T[j] ;
		}	
	}
	return (-1)*(sum / TrainDataSet.size());

};

NeuralNetw::~NeuralNetw(void)
{
}
