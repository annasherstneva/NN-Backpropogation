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

	z_exp = new double[Output_N];
	soft_max = new double[Output_N];

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
	int epoch = 0;
	while (epoch<15)
	{
		double ok = 0.0;
		Mix(TrainDataSet, Labels);
		std::cout << "-----------------------------------------------------------------" << std::endl;
		std::cout << "Number of ep = " << epoch << std::endl;
		
	/*	*/
		std::cout << "Calculating output and changing weights..." << std::endl;
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
			if (T[find_max(output, Output_N)] == 1.0)
				ok++;
			
			Check_grad(T);

			Change_Weights(GradO, GradH);
			Change_Delta(GradO, GradH);
		}
		std::cout << " Computing cross-entrophy..." << std::endl;
		double cross = 0.0;
		cross = CrossEntropy(TrainDataSet, Labels);
		std::cout << "Cross-entrophy: " << cross << " ";
		std::cout << std::endl;
		std::cout << "Number of right answers = " << ok<<std::endl;
		double error_calc = 0.0;
		error_calc = ok / TrainDataSet.size();
		std::cout << "Part of wrong answers = " <<(1- error_calc )<< std::endl;
		epoch++;
		if (cross <= 0.001)
		{
			break;
		}
		if (1 - error_calc <= 0.001)
		{
			break;
		}
		
	}
};

int NeuralNetw::find_max(double *mas, int size)
{
	double max = 0.0;
	int index = 0;
	for (int i = 0; i < size; i++)
	{
		if (max < mas[i])
		{
			max = mas[i];
			index = i;
		}
	}
	return index;
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
		return 1.0/(1.0+std::exp(-x));
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
		deltaOutput[j] = Get_random_number(-1.0,1.0);
	}
	for (int j = 0; j < Hide_N; j++)
	{
		deltaHide[j] = Get_random_number(-1.0, 1.0);
	}
};



void NeuralNetw::initialize_weights()
{
	
	
	for (int i = 0; i < Input_N; i++)
	{
		for (int j = 0; j < Hide_N; j++)
		{
			Hide_Weights[i][j] = Get_random_number(-1.0, 1.0);
		}
	}
	for (int i = 0; i < Hide_N; i++)
	{
		for (int j = 0; j < Output_N; j++)
		{
			Output_Weights[i][j] = Get_random_number(-1.0, 1.0);
		}
	}
};

double* NeuralNetw::softmax(double* sumOut)
{

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
			delta = speed_learning*Grad_o[j]*hide_output[i]*0.9;
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
			delta = speed_learning*Grad_o[j]*0.9;
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

void NeuralNetw::Mix(std::vector <std::vector <double>> Dataset, std::vector <double> Labels) 
{
	for (int i = 0; i<Dataset.size(); i++)
	{
		int nom1 = rand() % Dataset.size();
		int nom2 = rand() % Dataset.size();

		std::swap(Dataset[nom1], Dataset[nom2]);
		std::swap(Labels[nom1], Labels[nom2]);
	}
}

double NeuralNetw::Get_random_number(double min, double max)
{
	double fr = 1.0 /( (double)RAND_MAX + 0.1);
	return (double)(rand()*fr*(max - min) + min);
};


/*float NeuralNetw::Get_random_number() 
{
	float num = (float)dis(gen);
	float factor = (float)dis(gen);
	return factor < 0.5f ? -num : num;
}*/
NeuralNetw::~NeuralNetw(void)
{
}
